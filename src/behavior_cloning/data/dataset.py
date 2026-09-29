"""Dataset for demonstration trajectories.

Each sample contains a window of past observations and the corresponding future
actions in raw units.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

__all__ = ["MinMaxNormalizer", "NavigationDataset"]


class MinMaxNormalizer(nn.Module):
    """Min-max normalization to the range ``[-1, 1]``.

    Args:
        dim: Vector dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("lower", torch.zeros(dim))
        self.register_buffer("upper", torch.ones(dim))

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-6) -> "MinMaxNormalizer":
        """Return a normalizer whose bounds span ``(num_samples, dim)`` data.

        Args:
            data: Samples to take the per-dimension minimum and maximum of.
            eps: Minimum width of a dimension's range.
        """
        lower, upper = data.min(axis=0), data.max(axis=0)
        upper = np.where(upper - lower < eps, lower + eps, upper)
        normalizer = cls(data.shape[-1])
        normalizer.lower.copy_(torch.as_tensor(lower, dtype=torch.float32))
        normalizer.upper.copy_(torch.as_tensor(upper, dtype=torch.float32))
        return normalizer

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw values onto ``[-1, 1]``."""
        return (x - self.lower) / (self.upper - self.lower) * 2.0 - 1.0

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize`."""
        return (x + 1.0) / 2.0 * (self.upper - self.lower) + self.lower


class NavigationDataset(Dataset):
    """Dataset of observation histories and future action windows.

    Args:
        data_path: Path to the demo ``.npz`` file.
        obs_horizon: Number of past observations per sample.
        pred_horizon: Number of future actions per sample.
        num_episodes: Number of episodes to keep; ``None`` keeps all episodes.
    """

    def __init__(
        self,
        data_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_episodes: int | None = None,
    ) -> None:
        data = np.load(data_path)
        episode_ends = data["episode_ends"]
        if num_episodes is not None:
            episode_ends = episode_ends[:num_episodes]

        num_steps = int(episode_ends[-1])
        self.obs = data["obs"][:num_steps].astype(np.float32)
        self.action = data["action"][:num_steps].astype(np.float32)
        self.episode_ends = episode_ends
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)

        self.obs_index, self.action_index = self._build_indices()
        self.obs_normalizer = MinMaxNormalizer.fit(self.obs)
        self.action_normalizer = MinMaxNormalizer.fit(self.action)

    def _build_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-sample index arrays of shape ``(N, To)`` and ``(N, Tp)``."""
        starts = np.concatenate([[0], self.episode_ends[:-1]])
        anchors = np.arange(int(self.episode_ends[-1]))
        episode = np.searchsorted(self.episode_ends, anchors, side="right")
        lower, upper = starts[episode], self.episode_ends[episode] - 1

        obs_offsets = np.arange(-self.obs_horizon + 1, 1)
        action_offsets = np.arange(self.pred_horizon)
        obs_index = np.clip(anchors[:, None] + obs_offsets, lower[:, None], upper[:, None])
        action_index = np.clip(anchors[:, None] + action_offsets, lower[:, None], upper[:, None])
        return obs_index, action_index

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    def __len__(self) -> int:
        return len(self.obs_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.from_numpy(self.obs[self.obs_index[idx]]),
            "action": torch.from_numpy(self.action[self.action_index[idx]]),
        }

    def __repr__(self) -> str:
        return (
            f"NavigationDataset({len(self)} samples, {self.num_episodes} episodes, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"To={self.obs_horizon}, Tp={self.pred_horizon})"
        )
