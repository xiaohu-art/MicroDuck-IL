"""Checkpoint utilities for saving and restoring trained policies."""

import torch
from omegaconf import DictConfig, OmegaConf

from .policy import ChunkedPolicy
from .training import resolve_device

__all__ = ["load_checkpoint", "load_policy", "save_checkpoint"]


def save_checkpoint(path: str, policy: ChunkedPolicy, cfg: DictConfig, **extra) -> None:
    """Save a policy together with its config and metadata.

    Args:
        path: Destination checkpoint path.
        policy: Policy to save.
        cfg: Config used to build the policy.
        extra: Extra entries stored alongside the checkpoint.
    """
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "obs_dim": policy.head.cond_dim // policy.obs_horizon,
            "action_dim": policy.head.sample_dim // policy.pred_horizon,
            **extra,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """Return the raw checkpoint dictionary.

    Args:
        path: Checkpoint file.
        map_location: Device the tensors are read onto.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_policy(path: str, device: str = "auto", overrides: dict | None = None):
    """Load a policy from a checkpoint and restore its config.

    Args:
        path: Checkpoint file.
        device: Device name passed to :func:`~behavior_cloning.training.resolve_device`.
        overrides: Config overrides applied before reconstruction.

    Returns:
        The restored policy in evaluation mode and the config used to build it.
    """
    device = resolve_device(device)
    checkpoint = load_checkpoint(path, map_location=str(device))
    cfg = OmegaConf.create(checkpoint["config"])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    policy = ChunkedPolicy.from_config(cfg, checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy.to(device).eval(), cfg
