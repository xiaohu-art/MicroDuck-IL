"""Scripted expert that rounds the obstacle on a side drawn per episode."""

import torch

__all__ = ["ScriptedExpert"]


class ScriptedExpert:
    """Waypoint follower producing ``(vx, wz)`` velocity commands.

    Args:
        cfg: Composed config providing the ``task`` group.
        device: Device the commands are produced on.
    """

    def __init__(self, cfg, device: torch.device | str) -> None:
        env_cfg, expert_cfg = cfg.task.env, cfg.task.expert
        self.device = device
        self.detour_y = env_cfg.obstacle_half_w + expert_cfg.detour_margin
        self.detour_x = (
            env_cfg.obstacle_x - expert_cfg.detour_offset,
            env_cfg.obstacle_x + expert_cfg.detour_offset,
        )
        self.goal = torch.tensor(
            [env_cfg.obstacle_x + env_cfg.goal_offset, 0.0], dtype=torch.float32, device=device
        )
        self.waypoint_radius = expert_cfg.waypoint_radius
        self.v_max = expert_cfg.v_max
        self.w_max = expert_cfg.w_max
        self.k_heading = expert_cfg.k_heading

        self.branch = torch.empty(0, device=device)
        self.stage = torch.empty(0, dtype=torch.long, device=device)

    def reset(self, num_envs: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a side per episode and return it as ``+1`` (left) or ``-1`` (right)."""
        half = num_envs // 2
        branch = torch.cat([torch.ones(half), -torch.ones(num_envs - half)])
        order = torch.randperm(num_envs, generator=generator)
        self.branch = branch[order].to(self.device)
        self.stage = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        return self.branch

    def command(self, env) -> torch.Tensor:
        """Return ``(num_envs, 2)`` of ``(vx, wz)`` and advance the waypoint index.

        Args:
            env: A :class:`~behavior_cloning.env.navigation.NavigationEnv`.
        """
        position = env.env.base_pos[:, :2]
        yaw = env._yaw()

        corridor = [
            torch.stack([torch.full_like(self.branch, x), self.branch * self.detour_y], dim=-1)
            for x in self.detour_x
        ]
        waypoints = torch.stack(corridor + [self.goal.expand_as(position)])
        target = waypoints[self.stage, torch.arange(len(self.stage), device=self.device)]

        delta = target - position
        reached = delta.norm(dim=-1) < self.waypoint_radius
        self.stage[reached] = (self.stage[reached] + 1).clamp(max=len(waypoints) - 1)

        desired = torch.atan2(delta[:, 1], delta[:, 0])
        error = torch.atan2(torch.sin(desired - yaw), torch.cos(desired - yaw))
        yaw_rate = (self.k_heading * error).clamp(-self.w_max, self.w_max)
        forward = self.v_max * torch.cos(error).clamp(min=0.0)
        return torch.stack([forward, yaw_rate], dim=-1)
