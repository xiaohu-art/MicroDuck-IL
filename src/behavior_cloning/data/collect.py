"""Run the scripted expert and keep the episodes that reach the goal."""

import numpy as np
import torch

from .expert import ScriptedExpert

__all__ = ["collect_demonstrations"]


def collect_demonstrations(
    env, cfg, num_episodes: int, seed: int = 0, progress: bool = True
) -> dict[str, np.ndarray]:
    """Collect demonstrations in batches of ``env.num_envs``.

    Args:
        env: A :class:`~behavior_cloning.env.navigation.NavigationEnv`.
        cfg: Composed config providing the ``task`` group.
        num_episodes: Successful episodes to keep.
        seed: Offsets the per-batch start-pose seeds and the branch draw.
        progress: Print a line per batch.

    Returns:
        ``obs`` ``(steps, obs_dim)``, ``action`` ``(steps, 2)``,
        ``position`` ``(steps, 2)``, ``episode_ends`` ``(episodes,)`` and
        ``branch`` ``(episodes,)``.
    """
    expert = ScriptedExpert(cfg, env.device)
    generator = torch.Generator().manual_seed(seed)
    max_steps = cfg.task.env.max_steps

    obs_chunks, action_chunks, position_chunks = [], [], []
    episode_ends, branches, success_rates = [], [], []
    total_steps = 0
    batch = 0

    while len(episode_ends) < num_episodes:
        env.reset(seed=seed + batch)
        branch = expert.reset(env.num_envs, generator)

        observations, actions, positions, running = [], [], [], []
        for _ in range(max_steps):
            running.append((~env.done).clone())
            observations.append(env.observe().clone())
            positions.append(env.env.base_pos[:, :2].clone())
            command = expert.command(env)
            actions.append(command.clone())
            env.step(command)
            if env.done.all():
                break

        observations = torch.stack(observations).cpu().numpy()
        actions = torch.stack(actions).cpu().numpy()
        positions = torch.stack(positions).cpu().numpy()
        alive = torch.stack(running).cpu().numpy()
        success = env.success.cpu().numpy()
        success_rates.append(float(success.mean()))

        for index in range(env.num_envs):
            if len(episode_ends) >= num_episodes or not success[index]:
                continue
            steps = alive[:, index]
            obs_chunks.append(observations[steps, index])
            action_chunks.append(actions[steps, index])
            position_chunks.append(positions[steps, index])
            total_steps += int(steps.sum())
            episode_ends.append(total_steps)
            branches.append(float(branch[index]))

        batch += 1
        if progress:
            print(f"  batch {batch}: {len(episode_ends)}/{num_episodes} episodes, "
                  f"expert success {success_rates[-1]:.0%}")

    return {
        "obs": np.concatenate(obs_chunks).astype(np.float32),
        "action": np.concatenate(action_chunks).astype(np.float32),
        "position": np.concatenate(position_chunks).astype(np.float32),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
        "branch": np.asarray(branches, dtype=np.float32),
    }

