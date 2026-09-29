"""Diffusion-based action head with deterministic DDIM sampling."""

import math

import torch
from torch.nn import functional as F

from ..backbone import ConditionalMLP
from .base import ActionHead

__all__ = ["DiffusionHead", "make_beta_schedule"]


def make_beta_schedule(num_timesteps: int, schedule: str, max_beta: float = 0.999) -> torch.Tensor:
    """Return the per-step noise variances ``beta_t``, shape ``(num_timesteps,)``.

    Args:
        num_timesteps: Length of the forward process.
        schedule: ``squaredcos_cap_v2`` or ``linear``.
        max_beta: Upper bound on a single step's variance.
    """
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, num_timesteps)
    if schedule == "squaredcos_cap_v2":
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=max_beta).float()
    raise ValueError(f"Unknown beta schedule '{schedule}'")


class DiffusionHead(ActionHead):
    """Denoising diffusion head for flattened action chunks.

    Args:
        sample_dim: Flattened sample dimension.
        cond_dim: Conditioning dimension.
        num_train_timesteps: Number of forward-process steps.
        beta_schedule: Beta schedule name for :func:`make_beta_schedule`.
        prediction_type: Noise prediction type; only ``epsilon`` is supported.
        num_inference_steps: Default number of reverse steps.
        backbone: Keyword arguments forwarded to :class:`ConditionalMLP`.
    """

    def __init__(
        self,
        sample_dim: int,
        cond_dim: int,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        num_inference_steps: int = 100,
        **backbone,
    ) -> None:
        super().__init__(sample_dim, cond_dim)
        if prediction_type != "epsilon":
            raise ValueError(f"Only 'epsilon' prediction is implemented, got '{prediction_type}'")

        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(1.0 - make_beta_schedule(self.num_train_timesteps, beta_schedule), dim=0),
        )
        self.net = ConditionalMLP(
            out_dim=sample_dim, cond_dim=cond_dim, sample_dim=sample_dim, **backbone
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------
    def add_noise(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Return ``sqrt(alpha_bar_t) * sample + sqrt(1 - alpha_bar_t) * noise``.

        Args:
            sample: Clean samples ``(batch, sample_dim)``.
            noise: Standard normal noise of the same shape.
            timesteps: Integer steps ``(batch,)`` in ``[0, num_train_timesteps)``.
        """
        alpha_bar = self.alphas_cumprod[timesteps].unsqueeze(-1)
        return alpha_bar.sqrt() * sample + (1.0 - alpha_bar).sqrt() * noise

    def loss(self, sample: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Return the error on the noise added at a uniformly drawn timestep."""
        timesteps = torch.randint(
            0, self.num_train_timesteps, (sample.shape[0],), device=sample.device
        )
        noise = torch.randn_like(sample)
        noisy = self.add_noise(sample, noise, timesteps)
        prediction = self.net(
            cond=cond, sample=noisy, timestep=timesteps / self.num_train_timesteps
        )
        return F.mse_loss(prediction, noise)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------
    def _timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Return the strided subsequence of forward steps, from late to early."""
        steps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
        return steps.round().long().flip(0)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise standard normal noise into a sample ``(batch, sample_dim)``.

        Args:
            cond: Condition ``(batch, cond_dim)``, or None when unconditional.
            batch_size: Required only when unconditional.
            num_inference_steps: Reverse steps; defaults to the configured value.
        """
        batch_size = self._batch_size(cond, batch_size)
        timesteps = self._timesteps(num_inference_steps or self.num_inference_steps)

        x = torch.randn(batch_size, self.sample_dim, device=self._device())
        for index, t in enumerate(timesteps):
            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[index + 1]]
                if index + 1 < len(timesteps)
                else torch.ones_like(alpha_bar)
            )

            step = torch.full((batch_size,), t.item(), device=x.device)
            noise_pred = self.net(cond=cond, sample=x, timestep=step / self.num_train_timesteps)

            x0 = ((x - (1.0 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()).clamp(-1.0, 1.0)
            noise_pred = (x - alpha_bar.sqrt() * x0) / (1.0 - alpha_bar).sqrt()
            x = alpha_bar_prev.sqrt() * x0 + (1.0 - alpha_bar_prev).sqrt() * noise_pred
        return x
