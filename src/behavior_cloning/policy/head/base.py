"""Base interfaces for policy action heads."""

import torch
from torch import nn

__all__ = ["ActionHead"]


class ActionHead(nn.Module):
    """Base class for action heads.

    Args:
        sample_dim: Dimension of one flattened action sample.
        cond_dim: Dimension of the conditioning input.
    """

    def __init__(self, sample_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.sample_dim = int(sample_dim)
        self.cond_dim = int(cond_dim)

    def loss(self, sample: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Return the training loss for a batch of samples."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate one sample for each batch item."""
        raise NotImplementedError

    def _batch_size(self, cond: torch.Tensor | None, batch_size: int | None) -> int:
        """Return the batch size implied by the condition or given explicitly.

        Raises:
            ValueError: If neither is available.
        """
        if cond is not None:
            return cond.shape[0]
        if batch_size is None:
            raise ValueError("batch_size is required for an unconditional head")
        return batch_size

    def _device(self) -> torch.device:
        return next(self.parameters()).device
