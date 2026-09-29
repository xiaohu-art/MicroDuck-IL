"""MSE regression head for action prediction."""

import torch
from torch.nn import functional as F

from ..backbone import ConditionalMLP
from .base import ActionHead

__all__ = ["MSEHead"]


class MSEHead(ActionHead):
    """Regression head that predicts one action chunk per condition.

    Args:
        sample_dim: Dimension of one flattened action sample.
        cond_dim: Dimension of the conditioning input.
        backbone: Keyword arguments forwarded to :class:`ConditionalMLP`.
    """

    def __init__(self, sample_dim: int, cond_dim: int, **backbone) -> None:
        super().__init__(sample_dim, cond_dim)
        backbone.pop("time_embed_dim", None)
        self.net = ConditionalMLP(
            out_dim=sample_dim, cond_dim=cond_dim, sample_dim=0, time_embed_dim=0, **backbone
        )

    def loss(self, sample: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Return the MSE loss against the target sample."""
        prediction = self.net(cond=cond, batch_size=sample.shape[0])
        return F.mse_loss(prediction, sample)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Return the model prediction for the provided condition."""
        return self.net(cond=cond, batch_size=self._batch_size(cond, batch_size))
