"""Figures for the demonstrations and for the policies trained on them."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

__all__ = ["plot_demonstrations", "plot_toy_samples"]

LEFT_COLOR = "#c8451e"
RIGHT_COLOR = "#1f5fc4"


def _draw_scene(axis, cfg) -> None:
    """Draw the obstacle, the goal and the start marker."""
    env_cfg = cfg.task.env
    axis.add_patch(
        Rectangle(
            (env_cfg.obstacle_x - env_cfg.obstacle_depth / 2, -env_cfg.obstacle_half_w),
            env_cfg.obstacle_depth,
            2 * env_cfg.obstacle_half_w,
            color="0.35",
            zorder=3,
        )
    )
    goal_x = env_cfg.obstacle_x + env_cfg.goal_offset
    axis.add_patch(
        Circle((goal_x, 0.0), env_cfg.goal_radius, color="mediumseagreen", alpha=0.35, zorder=1)
    )
    axis.plot(0, 0, "k*", ms=14, zorder=5)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)


def plot_toy_samples(dataset, samples: dict, path: str, max_points: int = 2000) -> None:
    """Write the toy training data next to the generated samples.

    Args:
        dataset: A :class:`~behavior_cloning.data.toy.ToyDataset`.
        samples: Generated points ``(n, 2)`` keyed by the legend label.
        path: Output image path.
        max_points: Points drawn per group.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    data = dataset.sample[:max_points].numpy()
    axes[0].scatter(data[:, 0], data[:, 1], s=4, alpha=0.4, color="0.35")
    axes[0].set_title(f"data ({dataset.distribution})")

    for label, points in samples.items():
        points = points[:max_points]
        axes[1].scatter(points[:, 0], points[:, 1], s=4, alpha=0.4, label=label)
    axes[1].set_title("samples")
    if len(samples) > 1:
        axes[1].legend(markerscale=3, loc="upper right")

    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_demonstrations(data: dict, cfg, path: str, max_episodes: int = 60) -> None:
    """Write a top-down figure of the collected trajectories.

    Args:
        data: Arrays returned by
            :func:`~behavior_cloning.data.collect.collect_demonstrations`.
        cfg: Composed config providing the ``task`` group.
        path: Output image path.
        max_episodes: Trajectories drawn.
    """
    ends = data["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    branch = data["branch"]
    drawn = min(max_episodes, len(ends))

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    _draw_scene(axis, cfg)
    for index in range(drawn):
        path_xy = data["position"][starts[index] : ends[index]]
        color = LEFT_COLOR if branch[index] > 0 else RIGHT_COLOR
        axis.plot(path_xy[:, 0], path_xy[:, 1], color=color, lw=1.2, alpha=0.8, zorder=4)
    axis.set_title(f"{drawn} demonstrations, left / right")

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
