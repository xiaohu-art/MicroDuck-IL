"""Conditional MLP backbone used by action heads.

The network combines optional sample, condition, and timestep inputs into a
single feed-forward output.
"""

import math

import torch
from torch import nn

__all__ = ["ConditionalMLP", "SinusoidalTimeEmbedding", "resolve_activation"]

_TIME_SCALE = 1000.0

_ACTIVATIONS = {
    "mish": nn.Mish,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def resolve_activation(name: str) -> nn.Module:
    """Return the activation module assigned to the provided name.

    Raises:
        ValueError: If the name is not recognized.
    """
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}', expected one of {list(_ACTIVATIONS)}")


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar timestep values.

    Args:
        dim: Embedding width. Must be even.
        max_period: Ratio between the lowest and highest frequency.

    Raises:
        ValueError: If ``dim`` is odd.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be even, got {dim}")
        self.dim = dim
        half = dim // 2
        self.register_buffer(
            "frequencies", torch.exp(-math.log(max_period) * torch.arange(half) / half)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed times ``(batch,)`` into ``(batch, dim)``."""
        angles = t.float().reshape(-1, 1) * _TIME_SCALE * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConditionalMLP(nn.Module):
    """MLP over optional sample, condition, and timestep inputs.

    Args:
        out_dim: Output dimension.
        cond_dim: Condition dimension. Use 0 for unconditional models.
        hidden_dims: Hidden-layer widths.
        activation: Activation name accepted by :func:`resolve_activation`.
        sample_dim: Sample dimension. Use 0 when no sample input is provided.
        time_embed_dim: Time embedding dimension. Use 0 when no timestep input is provided.
    """

    def __init__(
        self,
        out_dim: int,
        cond_dim: int,
        hidden_dims=(512, 512, 512),
        activation: str = "mish",
        sample_dim: int = 0,
        time_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.sample_dim = int(sample_dim)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.time_embedding = (
            SinusoidalTimeEmbedding(self.time_embed_dim) if self.time_embed_dim else None
        )

        in_dim = self.sample_dim + self.cond_dim + self.time_embed_dim
        dims = [in_dim or 1, *hidden_dims]   # a constant input when every width is zero
        layers: list[nn.Module] = []
        for in_dim, hidden_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, hidden_dim), resolve_activation(activation)]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        cond: torch.Tensor | None = None,
        sample: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Map the present inputs to ``(batch, out_dim)``.

        Args:
            cond: Condition ``(batch, cond_dim)``, or None when unconditional.
            sample: Sample ``(batch, sample_dim)``, or None for the regression head.
            timestep: Times ``(batch,)`` on ``[0, 1]``, or None without a time input.
            batch_size: Required only when every input width is zero.
        """
        parts = []
        if self.sample_dim:
            parts.append(sample)
        if self.cond_dim:
            parts.append(cond)
        if self.time_embedding is not None:
            parts.append(self.time_embedding(timestep))
        if parts:
            return self.net(torch.cat(parts, dim=-1))

        device = next(self.parameters()).device
        return self.net(torch.ones(batch_size, 1, device=device))
