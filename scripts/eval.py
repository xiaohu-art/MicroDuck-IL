"""Evaluate the policy from the newest checkpoint.

    uv run python scripts/eval.py
    uv run python scripts/eval.py --checkpoint outputs/flow_s0/policy.pt
    uv run python scripts/eval.py --nfe 10 --action-horizon 1
"""

import argparse
import glob
import os
import time

import genesis as gs
import numpy as np
import torch

from behavior_cloning.checkpoint import load_policy
from behavior_cloning.env import NavigationEnv
from behavior_cloning.plots import plot_evaluation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUCCESS, COLLISION, FELL, TIMEOUT = 0, 1, 2, 3


def find_latest_checkpoint(root: str) -> str:
    """Return the most recently written ``policy.pt`` under a directory.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    candidates = glob.glob(os.path.join(root, "**", "policy.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No policy.pt found under '{root}'")
    return max(candidates, key=os.path.getmtime)


def rollout(env: NavigationEnv, policy, cfg) -> dict:
    """Run one batch of episodes and return the recorded trajectories.

    Args:
        env: Navigation environment sized to the number of episodes.
        policy: Policy producing ``(num_envs, Ta, 2)`` action chunks.
        cfg: Config providing the ``task`` and ``eval`` groups.

    Returns:
        ``position`` ``(steps, episodes, 2)``, ``command`` ``(steps, episodes, 2)``,
        ``alive`` ``(steps, episodes)`` and ``outcome`` ``(episodes,)``.
    """
    observation = env.reset(seed=cfg.eval.scenario_seed)
    torch.manual_seed(cfg.eval.sample_seed)
    history = [observation] * policy.obs_horizon

    positions, commands, alive = [], [], []
    for _ in range(cfg.task.env.max_steps):
        chunk = policy.predict_action(torch.stack(history[-policy.obs_horizon :], dim=1))
        for index in range(policy.action_horizon):
            if env.done.all():
                break
            command = chunk[:, index]
            alive.append((~env.done).clone())
            positions.append(env.env.base_pos[:, :2].clone())
            commands.append(command.clone())
            history.append(env.step(command))
        if env.done.all():
            break

    outcome = torch.full((env.num_envs,), TIMEOUT, dtype=torch.long, device=env.device)
    outcome[env.done] = COLLISION
    outcome[env.fell] = FELL
    outcome[env.success] = SUCCESS
    return {
        "position": torch.stack(positions).cpu().numpy(),
        "command": torch.stack(commands).cpu().numpy(),
        "alive": torch.stack(alive).cpu().numpy(),
        "outcome": outcome.cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a behavior-cloning policy.")
    parser.add_argument("--checkpoint", help="defaults to the newest one under outputs/")
    parser.add_argument("--episodes", type=int, help="overrides eval.num_episodes")
    parser.add_argument("--nfe", type=int, help="overrides the head's inference steps")
    parser.add_argument("--action-horizon", type=int, help="overrides Ta")
    parser.add_argument("--device", default="cpu", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint(os.path.join(REPO_ROOT, "outputs"))
    overrides = {}
    if args.nfe is not None:
        overrides["policy"] = {"num_inference_steps": args.nfe}
    if args.action_horizon is not None:
        overrides["task"] = {"action_horizon": args.action_horizon}

    policy, cfg = load_policy(checkpoint, device=args.device, overrides=overrides)
    episodes = args.episodes or cfg.eval.num_episodes
    print(f"checkpoint {checkpoint}")
    print(
        f"  head={cfg.policy.head}  seed={cfg.seed}  Ta={policy.action_horizon}"
        f"  nfe={getattr(policy.head, 'num_inference_steps', '-')}  episodes={episodes}"
    )

    gs.init(backend=gs.cpu, seed=cfg.seed, logging_level="warning")
    env = NavigationEnv(cfg, num_envs=episodes)

    start = time.time()
    data = rollout(env, policy, cfg)
    elapsed = time.time() - start

    counts = np.bincount(data["outcome"], minlength=4)
    print(f"\n  success   {counts[SUCCESS] / episodes:.0%}  ({counts[SUCCESS]}/{episodes})")
    print(f"  collision {counts[COLLISION]}\n  fell      {counts[FELL]}\n  timeout   {counts[TIMEOUT]}")
    print(f"  {elapsed:.1f} s on {args.device}")

    figure = os.path.join(os.path.dirname(checkpoint), "evaluation.png")
    plot_evaluation(data, cfg, figure)
    print(f"  figure {figure}")


if __name__ == "__main__":
    main()
