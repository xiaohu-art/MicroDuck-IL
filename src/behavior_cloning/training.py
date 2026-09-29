"""Helpers for seeding, optimization, scheduling, and checkpoint-safe EMA."""

import copy
import math
import random

import numpy as np
import torch

__all__ = ["EMA", "make_optimizer", "make_scheduler", "resolve_device", "set_seed"]


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and torch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Return the requested device.

    Args:
        name: ``auto``, ``cpu``, ``cuda`` or ``mps``; ``auto`` takes the fastest available.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Return an AdamW optimizer over the model parameters.

    Args:
        model: Model to optimize.
        cfg: The ``train`` config group providing ``learning_rate`` and ``weight_decay``.
    """
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, num_steps: int, num_warmup_steps: int = 0
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a linear warmup followed by a cosine decay to zero.

    Args:
        optimizer: Optimizer whose learning rate is scaled.
        num_steps: Total optimizer steps.
        num_warmup_steps: Steps spent ramping up from zero.
    """

    def factor(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Exponential moving average of a model's parameters and buffers.

    Args:
        model: Model to track.
        decay: Weight kept from the running average on each update.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Blend the model's current tensors into the average."""
        for average, current in zip(self.model.state_dict().values(), model.state_dict().values()):
            if average.dtype.is_floating_point:
                average.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                average.copy_(current)
