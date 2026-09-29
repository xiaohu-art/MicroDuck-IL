"""2D toy distributions and scoring utilities for policy evaluation."""

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ToyDataset", "toy_metrics"]

RADIUS = 0.8
MODE_BASE = np.pi / 4
MODE_SPAN = np.pi / 2


class ToyDataset(Dataset):
    """Dataset for 2D toy distributions with optional conditioning.

    Args:
        distribution: ``two_moons`` or ``rotating_modes``.
        num_samples: Number of samples to draw.
        noise: Standard deviation added to each coordinate.
        seed: Random seed used for sampling.

    Raises:
        ValueError: If the distribution name is unknown.
    """

    def __init__(
        self,
        distribution: str = "two_moons",
        num_samples: int = 16384,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        if distribution == "two_moons":
            sample, cond = self._two_moons(rng, num_samples)
        elif distribution == "rotating_modes":
            sample, cond = self._rotating_modes(rng, num_samples)
        else:
            raise ValueError(f"Unknown distribution '{distribution}'")

        sample = sample + rng.normal(0.0, noise, sample.shape)
        self.distribution = distribution
        self.sample = torch.as_tensor(sample, dtype=torch.float32)
        self.cond = torch.as_tensor(cond, dtype=torch.float32)

    @staticmethod
    def _two_moons(rng, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate a two-moon dataset in normalized coordinates."""
        half = num_samples // 2
        outer_t = rng.uniform(0.0, np.pi, half)
        inner_t = rng.uniform(0.0, np.pi, num_samples - half)
        outer = np.stack([np.cos(outer_t), np.sin(outer_t)], axis=-1)
        inner = np.stack([1.0 - np.cos(inner_t), 0.5 - np.sin(inner_t)], axis=-1)
        sample = (np.concatenate([outer, inner]) - [0.5, 0.25]) / 1.5 * RADIUS
        return sample[rng.permutation(num_samples)], np.zeros((num_samples, 0))

    @staticmethod
    def _rotating_modes(rng, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate two circular modes whose angle depends on the condition."""
        cond = rng.uniform(0.0, 1.0, num_samples)
        sign = rng.integers(0, 2, num_samples) * 2 - 1
        angle = sign * (MODE_BASE + cond * MODE_SPAN)
        sample = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * RADIUS
        return sample, cond[:, None]

    @property
    def cond_dim(self) -> int:
        return self.cond.shape[-1]

    def __len__(self) -> int:
        return len(self.sample)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"sample": self.sample[idx], "cond": self.cond[idx]}

    def __repr__(self) -> str:
        return f"ToyDataset({self.distribution}, {len(self)} samples, cond_dim={self.cond_dim})"


def _moon_manifold(num_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return noise-free crescent points and the crescent index of each."""
    t = np.linspace(0.0, np.pi, num_points)
    outer = np.stack([np.cos(t), np.sin(t)], axis=-1)
    inner = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=-1)
    points = (np.concatenate([outer, inner]) - [0.5, 0.25]) / 1.5 * RADIUS
    return points, np.repeat([0, 1], num_points)


def toy_metrics(distribution: str, samples: np.ndarray, cond_value: float | None = None) -> dict:
    """Score generated samples against the target toy manifold.

    Args:
        distribution: Distribution name.
        samples: Generated points with shape ``(num_samples, 2)``.
        cond_value: Conditioning value used for ``rotating_modes``.

    Returns:
        A dictionary with coverage and manifold-distance metrics.

    Raises:
        ValueError: If the distribution name is unknown.
    """
    if distribution == "two_moons":
        points, labels = _moon_manifold()
        distances = np.linalg.norm(samples[:, None, :] - points[None], axis=-1)
        cluster = labels[distances.argmin(axis=1)]
        return {
            "coverage": float(min((cluster == 0).mean(), (cluster == 1).mean())),
            "off_manifold": float(distances.min(axis=1).mean()),
        }

    if distribution == "rotating_modes":
        angle = np.arctan2(samples[:, 1], samples[:, 0])
        deviation = np.abs(np.abs(angle) - (MODE_BASE + cond_value * MODE_SPAN))
        return {
            "coverage": float(min((angle > 0).mean(), (angle < 0).mean())),
            "off_manifold": float(np.minimum(deviation, 2 * np.pi - deviation).mean()),
            "mean_angle": float(np.abs(angle).mean()),
        }

    raise ValueError(f"Unknown distribution '{distribution}'")
