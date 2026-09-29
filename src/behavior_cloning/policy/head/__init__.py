from omegaconf import DictConfig, OmegaConf

from .base import ActionHead
from .diffusion import DiffusionHead
from .flow import FlowHead
from .mse import MSEHead

__all__ = ["ActionHead", "DiffusionHead", "FlowHead", "MSEHead", "HEADS", "build_head"]

HEADS = {"mse": MSEHead, "diffusion": DiffusionHead, "flow": FlowHead}


def build_head(cfg, sample_dim: int, cond_dim: int) -> ActionHead:
    """Construct the action head selected by configuration.

    Args:
        cfg: Policy config with a ``head`` entry and optional backbone settings.
        sample_dim: Flattened sample dimension.
        cond_dim: Conditioning dimension.

    Raises:
        ValueError: If the head name is not registered.
    """
    kwargs = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    name = kwargs.pop("head")
    kwargs.update(kwargs.pop("backbone", {}))
    if name not in HEADS:
        raise ValueError(f"Unknown head '{name}', expected one of {sorted(HEADS)}")
    return HEADS[name](sample_dim=sample_dim, cond_dim=cond_dim, **kwargs)
