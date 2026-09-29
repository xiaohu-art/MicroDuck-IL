"""Flow matching head for rectified-flow action generation."""

import torch
from torch.nn import functional as F

from ..backbone import ConditionalMLP
from .base import ActionHead

__all__ = ["FlowHead"]


class FlowHead(ActionHead):
    """Rectified-flow head for flattened action chunks.

    Args:
        sample_dim: Flattened sample dimension.
        cond_dim: Conditioning dimension.
        num_inference_steps: Default number of Euler integration steps.
        backbone: Keyword arguments forwarded to :class:`ConditionalMLP`.
    """

    def __init__(
        self, sample_dim: int, cond_dim: int, num_inference_steps: int = 10, **backbone
    ) -> None:
        super().__init__(sample_dim, cond_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.net = ConditionalMLP(
            out_dim=sample_dim, cond_dim=cond_dim, sample_dim=sample_dim, **backbone
        )

    def loss(self, sample: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Return the error on the path velocity at a uniformly drawn time."""
        noise = torch.randn_like(sample)
        t = torch.rand(sample.shape[0], device=sample.device)
        interpolated = (1.0 - t).unsqueeze(-1) * noise + t.unsqueeze(-1) * sample
        target = sample - noise
        return F.mse_loss(self.net(cond=cond, sample=interpolated, timestep=t), target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate the velocity field from noise to a sample ``(batch, sample_dim)``.

        Args:
            cond: Condition ``(batch, cond_dim)``, or None when unconditional.
            batch_size: Required only when unconditional.
            num_inference_steps: Euler steps; defaults to the configured value.
        """
        batch_size = self._batch_size(cond, batch_size)
        steps = num_inference_steps or self.num_inference_steps
        dt = 1.0 / steps

        x = torch.randn(batch_size, self.sample_dim, device=self._device())
        for index in range(steps):
            t = torch.full((batch_size,), index * dt, device=x.device)
            x = x + self.net(cond=cond, sample=x, timestep=t) * dt
        return x
