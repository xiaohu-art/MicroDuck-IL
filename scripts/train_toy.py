"""Fit a 2-D toy distribution with one action head.

    uv run python scripts/train_toy.py policy=diffusion
    uv run python scripts/train_toy.py policy=mse toy.distribution=rotating_modes
"""

import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from behavior_cloning.data import ToyDataset, toy_metrics
from behavior_cloning.logger import Logger
from behavior_cloning.plots import plot_toy_samples
from behavior_cloning.policy import build_head
from behavior_cloning.training import make_optimizer, make_scheduler, resolve_device, set_seed


def draw_samples(head, num_samples: int, cond_dim: int, cond_value, device) -> np.ndarray:
    """Draw samples at one condition value and return them as ``(num_samples, 2)``.

    Args:
        head: Trained action head.
        num_samples: Points drawn.
        cond_dim: Condition width; 0 draws unconditionally.
        cond_value: Scalar condition, unused when ``cond_dim`` is 0.
        device: Device the head runs on.
    """
    cond = None
    if cond_dim:
        cond = torch.full((num_samples, cond_dim), float(cond_value), device=device)
    return head.sample(cond=cond, batch_size=num_samples).cpu().numpy()


@hydra.main(version_base="1.3", config_path="../configs", config_name="toy")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = HydraConfig.get().runtime.output_dir

    dataset = ToyDataset(**cfg.toy, seed=cfg.seed)
    sample = dataset.sample.to(device)
    cond = dataset.cond.to(device)
    print(f"{dataset}  head={cfg.policy.head}  device={device}")

    head = build_head(cfg.policy, sample_dim=sample.shape[-1], cond_dim=dataset.cond_dim).to(device)
    optimizer = make_optimizer(head, cfg.train)
    scheduler = make_scheduler(optimizer, cfg.train.num_steps, cfg.train.lr_warmup_steps)
    logger = Logger(os.path.join(output_dir, "train_log.csv"), total=cfg.train.num_steps)

    running = 0.0
    for step in range(1, cfg.train.num_steps + 1):
        index = torch.randint(0, len(dataset), (cfg.train.batch_size,), device=device)
        loss = head.loss(sample[index], cond[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if step % cfg.train.log_interval == 0:
            logger.log(step, loss=running / cfg.train.log_interval, lr=scheduler.get_last_lr()[0])
            running = 0.0
    logger.close()

    head.eval()
    cond_values = list(cfg.eval.cond_values) if dataset.cond_dim else [None]
    generated, metrics = {}, []
    for value in cond_values:
        points = draw_samples(head, cfg.eval.num_samples, dataset.cond_dim, value, device)
        generated["samples" if value is None else f"c={value:.1f}"] = points
        metrics.append(toy_metrics(dataset.distribution, points, value))

    print(f"\n{'':<10}{'coverage':>12}{'off-manifold':>14}")
    for label, row in zip(generated, metrics):
        print(f"{label:<10}{row['coverage']:12.3f}{row['off_manifold']:14.3f}")
    if len(metrics) > 1:
        coverage = float(np.mean([row["coverage"] for row in metrics]))
        off_manifold = float(np.mean([row["off_manifold"] for row in metrics]))
        print(f"{'mean':<10}{coverage:12.3f}{off_manifold:14.3f}")
    if dataset.cond_dim:
        angles = [row["mean_angle"] for row in metrics]
        print(f"\ncondition following {np.corrcoef(cond_values, angles)[0, 1]:.3f}")

    figure = os.path.join(output_dir, "toy_samples.png")
    plot_toy_samples(dataset, generated, figure)
    print(f"figure {figure}")


if __name__ == "__main__":
    main()
