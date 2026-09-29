"""Collect demonstrations with the scripted expert.

    uv run python scripts/collect.py
    uv run python scripts/collect.py task.collect.num_episodes=100
"""

import os

import genesis as gs
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from behavior_cloning.data import collect_demonstrations
from behavior_cloning.env import NavigationEnv
from behavior_cloning.plots import plot_demonstrations

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    gs.init(backend=gs.cpu, seed=cfg.seed, logging_level="warning")
    env = NavigationEnv(cfg, num_envs=cfg.task.collect.batch_size)

    data = collect_demonstrations(env, cfg, cfg.task.collect.num_episodes, seed=cfg.seed)

    path = os.path.join(REPO_ROOT, cfg.task.data_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)

    lengths = np.diff(np.concatenate([[0], data["episode_ends"]]))
    left = int((data["branch"] > 0).sum())
    print(f"\n{path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
    print(f"  {len(data['episode_ends'])} episodes, {len(data['obs'])} steps")
    print(f"  length min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")
    print(f"  left {left} / right {len(data['branch']) - left}")

    figure = os.path.join(HydraConfig.get().runtime.output_dir, "demonstrations.png")
    plot_demonstrations(data, cfg, figure)
    print(f"  figure {figure}")


if __name__ == "__main__":
    main()
