"""Obstacle-avoidance navigation on top of a frozen locomotion policy.

    high-level policy  -- (vx, wz) at 5 Hz -->  frozen PPO actor at 50 Hz  -->  joints

The duck starts facing a wall that spans its path and must reach a goal behind it.
"""

import genesis as gs
import torch

from ..microduck.algorithm import MLPModel
from ..microduck.env import Env

__all__ = ["NavigationEnv"]


class _DuckEnv(Env):
    """MicroDuck environment with a static obstacle added before the scene is built.

    Args:
        env_cfg: The ``lowlevel.env`` config group.
        robot_cfg: The ``lowlevel.robot`` config group.
        obstacle: Morph added to the scene alongside the robot.
        show_viewer: Open the interactive viewer.
    """

    def __init__(self, env_cfg, robot_cfg, obstacle, show_viewer=False) -> None:
        self._obstacle = obstacle
        super().__init__(env_cfg, robot_cfg, show_viewer=show_viewer)

    def _init_scene(self, show_viewer: bool = False, record_video: bool = False) -> None:
        """Create the scene, add ground, robot and obstacle, then build it."""
        from ..microduck.assets import resolve_model_path

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.2, -1.2, 0.9),
                camera_lookat=(0.6, 0.0, 0.1),
                camera_fov=45,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=str(resolve_model_path(self.robot_cfg.model_path)))
        )
        self.obstacle = self.scene.add_entity(self._obstacle)
        self.scene.build(n_envs=self.num_envs, env_spacing=(4.0, 4.0))


class NavigationEnv:
    """Batched obstacle-avoidance episodes driven by velocity commands.

    Args:
        cfg: Composed config providing the ``task`` and ``lowlevel`` groups.
        num_envs: Episodes to run in parallel; defaults to ``lowlevel.env.sim.num_envs``.
    """

    def __init__(self, cfg, num_envs: int | None = None) -> None:
        self.cfg = cfg
        env_cfg = cfg.task.env
        if num_envs is not None:
            cfg.lowlevel.env.sim.num_envs = num_envs

        obstacle = gs.morphs.Box(
            pos=(env_cfg.obstacle_x, 0.0, env_cfg.obstacle_height / 2),
            size=(env_cfg.obstacle_depth, 2 * env_cfg.obstacle_half_w, env_cfg.obstacle_height),
            fixed=True,
        )
        self.env = _DuckEnv(cfg.lowlevel.env, cfg.lowlevel.robot, obstacle)
        observation = self.env.reset()

        self.actor = MLPModel(
            observation, cfg.lowlevel.obs_groups, "actor", self.env.action_dim, **cfg.lowlevel.actor
        ).to(self.device)
        loaded = torch.load(
            cfg.task.lowlevel_checkpoint, map_location=self.device, weights_only=False
        )
        self.actor.load_state_dict(loaded["actor_state_dict"])
        self.actor.eval()
        self._observation = observation

        self.goal = torch.tensor(
            [env_cfg.obstacle_x + env_cfg.goal_offset, 0.0], dtype=gs.tc_float, device=self.device
        )
        self._obstacle_half = (   # inflated by the duck's radius
            env_cfg.obstacle_depth / 2 + 0.06,
            env_cfg.obstacle_half_w + 0.06,
        )
        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device

    @property
    def obs_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def step_dt(self) -> float:
        """Seconds per high-level step."""
        return self.env.step_dt * self.cfg.task.env.decimation

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _yaw(self) -> torch.Tensor:
        """Return the base yaw angle, shape ``(num_envs,)``."""
        w, x, y, z = self.env.base_quat.unbind(dim=-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def observe(self) -> torch.Tensor:
        """Return the high-level observation, shape ``(num_envs, 8)``."""
        position = self.env.base_pos[:, :2]
        yaw = self._yaw()
        cos, sin = torch.cos(-yaw), torch.sin(-yaw)
        to_goal = self.goal[None] - position

        goal_in_base = torch.stack(
            [cos * to_goal[:, 0] - sin * to_goal[:, 1], sin * to_goal[:, 0] + cos * to_goal[:, 1]],
            dim=-1,
        )
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        linear_velocity = self.env.base_lin_vel[:, :2]
        yaw_rate = self.env.base_ang_vel[:, 2:3]
        obstacle_ahead = self.cfg.task.env.obstacle_x - position[:, 0:1]

        return torch.cat(
            [goal_in_base, heading, linear_velocity, yaw_rate, obstacle_ahead], dim=-1
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Restart every episode and return the first observation.

        Args:
            seed: Draws the start poses; None uses the nominal pose.
        """
        self.env.reset()
        if seed is not None:
            env_cfg = self.cfg.task.env
            generator = torch.Generator().manual_seed(seed)
            n = self.num_envs
            offset = torch.zeros(n, 3)
            offset[:, 1] = (torch.rand(n, generator=generator) * 2 - 1) * env_cfg.start_y
            yaw = (torch.rand(n, generator=generator) * 2 - 1) * env_cfg.start_yaw
            quaternion = torch.stack(
                [torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], dim=-1
            )
            position = self.env.init_base_pos.cpu().unsqueeze(0).repeat(n, 1) + offset
            envs_idx = torch.arange(n, device=self.device)
            self.env.robot.set_pos(position.to(self.device), envs_idx=envs_idx, zero_velocity=True)
            self.env.robot.set_quat(quaternion.to(self.device), envs_idx=envs_idx, zero_velocity=True)
            self.env._update_robot_state()

        self._observation = self.env.get_observations()

        n, device = self.num_envs, self.device
        self.step_count = 0
        self.done = torch.zeros(n, dtype=torch.bool, device=device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.collided = torch.zeros(n, dtype=torch.bool, device=device)
        self.fell = torch.zeros(n, dtype=torch.bool, device=device)
        return self.observe()

    def _update_done(self) -> None:
        """Latch the outcome of every episode that ends this step."""
        position = self.env.base_pos[:, :2]
        inside = ((position[:, 0] - self.cfg.task.env.obstacle_x).abs() < self._obstacle_half[0]) & (
            position[:, 1].abs() < self._obstacle_half[1]
        )
        reached = (position - self.goal[None]).norm(dim=-1) < self.cfg.task.env.goal_radius
        tilt = torch.acos((-self.env.projected_gravity[:, 2]).clamp(-1.0, 1.0))
        fell = tilt > 1.22  # 70 deg

        running = ~self.done
        self.collided |= inside & running
        self.reached |= reached & running
        self.fell |= fell & running
        self.done |= inside | reached | fell

    @torch.no_grad()
    def step(self, command: torch.Tensor) -> torch.Tensor:
        """Hold one velocity command for a full high-level step.

        Args:
            command: ``(num_envs, 2)`` of ``(vx, wz)`` in the base frame.

        Returns:
            The high-level observation after the step.
        """
        velocity = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=self.device)
        velocity[:, 0] = command[:, 0]
        velocity[:, 2] = command[:, 1]

        command_term = self.env.command_term
        command_term.time_left[:] = 1e6  # suppress the low-level resampling timer
        for _ in range(self.cfg.task.env.decimation):
            command_term.vel_command_b[:] = velocity
            self._observation, _, _, _ = self.env.step(self.actor(self._observation))

        self.step_count += 1
        self._update_done()
        return self.observe()

    @property
    def success(self) -> torch.Tensor:
        """Episodes that reached the goal without colliding or falling."""
        return self.reached & ~self.collided & ~self.fell
