"""Action-chunking policy wrapping an action head."""

import copy

import torch
from torch import nn

from ..data.dataset import MinMaxNormalizer
from .head import build_head

__all__ = ["ChunkedPolicy"]


class ChunkedPolicy(nn.Module):
    """Action-chunking policy that wraps a normalized action head.

    Args:
        head: Action head over flattened chunks.
        obs_normalizer: Normalizer for the observation history.
        action_normalizer: Normalizer for the action chunk.
        obs_horizon: Number of observations in each conditioning window.
        pred_horizon: Number of actions predicted per chunk.
        action_horizon: Number of actions executed from the predicted chunk.
    """

    def __init__(
        self,
        head,
        obs_normalizer,
        action_normalizer,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
    ) -> None:
        super().__init__()
        self.head = head
        self.obs_normalizer = copy.deepcopy(obs_normalizer)
        self.action_normalizer = copy.deepcopy(action_normalizer)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.action_horizon = int(action_horizon)

    @classmethod
    def from_config(
        cls,
        cfg,
        obs_dim: int,
        action_dim: int,
        obs_normalizer=None,
        action_normalizer=None,
    ) -> "ChunkedPolicy":
        """Construct a policy from a configuration object.

        Args:
            cfg: Config containing ``policy`` and ``task`` groups.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            obs_normalizer: Optional observation normalizer.
            action_normalizer: Optional action normalizer.
        """
        task = cfg.task
        head = build_head(
            cfg.policy,
            sample_dim=task.pred_horizon * action_dim,
            cond_dim=task.obs_horizon * obs_dim,
        )
        return cls(
            head,
            obs_normalizer if obs_normalizer is not None else MinMaxNormalizer(obs_dim),
            action_normalizer if action_normalizer is not None else MinMaxNormalizer(action_dim),
            task.obs_horizon,
            task.pred_horizon,
            task.action_horizon,
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the head loss for a batch of ``obs`` and ``action`` in raw units."""
        cond = self.obs_normalizer.normalize(batch["obs"]).flatten(1)
        sample = self.action_normalizer.normalize(batch["action"]).flatten(1)
        return self.head.loss(sample, cond)

    @torch.no_grad()
    def predict_chunk(self, obs: torch.Tensor, **sample_kwargs) -> torch.Tensor:
        """Map ``(batch, To, obs_dim)`` to ``(batch, Tp, action_dim)``, both in raw units.

        Args:
            obs: Observation history.
            sample_kwargs: Forwarded to the head's sampler.
        """
        cond = self.obs_normalizer.normalize(obs).flatten(1)
        sample = self.head.sample(cond=cond, **sample_kwargs)
        chunk = sample.reshape(obs.shape[0], self.pred_horizon, -1)
        return self.action_normalizer.unnormalize(chunk)

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor, **sample_kwargs) -> torch.Tensor:
        """Return the first ``Ta`` actions of the predicted chunk."""
        return self.predict_chunk(obs, **sample_kwargs)[:, : self.action_horizon]
