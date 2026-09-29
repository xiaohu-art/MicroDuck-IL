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
        timesteps = torch.rand(sample.shape[0], device=sample.device)
        noise = torch.randn_like(sample)

        # TODO (4/5): Form `x_t` and the target velocity:
        #   x_t = (1 - timesteps[:, None]) * noise + timesteps[:, None] * sample
        #   target = sample - noise
        # Predict `target` with `self.net(cond=cond, sample=x_t, timestep=timesteps)`
        # and return `F.mse_loss(prediction, target)`.
        raise NotImplementedError

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
        x = torch.randn(batch_size, self.sample_dim, device=self._device())
        step_size = 1.0 / steps

        # TODO (5/5): Integrate the velocity field with Euler's method:
        #   for i in range(steps):
        #       t = torch.full((batch_size,), i * step_size, device=x.device)
        #       velocity = self.net(cond=cond, sample=x, timestep=t)
        #       x = x + step_size * velocity
        # Return `x` after the loop.
        raise NotImplementedError
