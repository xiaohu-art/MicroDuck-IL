"""Figures for the demonstrations and for the policies trained on them."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

__all__ = ["plot_demonstrations", "plot_evaluation", "plot_toy_samples"]

LEFT_COLOR = "#c8451e"
RIGHT_COLOR = "#1f5fc4"

# Episode outcomes, ordered as encoded by scripts/eval.py.
OUTCOMES = [
    ("success", "mediumseagreen"),
    ("collision", "#c8451e"),
    ("fell", "0.55"),
    ("timeout", "#e08a1e"),
]


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
    axis.set_xlim(-0.2, goal_x + 0.3)
    axis.set_ylim(-0.7, 0.7)          # the detour corridor reaches +/-0.5
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


def plot_evaluation(data: dict, cfg, path: str) -> None:
    """Write the evaluation rollouts next to the commands that produced them.

    Args:
        data: ``position`` ``(steps, episodes, 2)``, ``command`` ``(steps, episodes, 2)``,
            ``alive`` ``(steps, episodes)`` and ``outcome`` ``(episodes,)`` indexing
            :data:`OUTCOMES`.
        cfg: Composed config providing the ``task`` group.
        path: Output image path.
    """
    outcome = data["outcome"]
    alive = data["alive"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

    _draw_scene(axes[0], cfg)
    for episode in range(len(outcome)):
        steps = alive[:, episode]
        color = OUTCOMES[outcome[episode]][1]
        axes[0].plot(
            data["position"][steps, episode, 0],
            data["position"][steps, episode, 1],
            color=color,
            lw=1.1,
            alpha=0.85,
            zorder=4,
        )
    counts = np.bincount(outcome, minlength=len(OUTCOMES))
    axes[0].legend(
        handles=[
            Line2D([], [], color=color, lw=2, label=f"{name} {counts[index]}")
            for index, (name, color) in enumerate(OUTCOMES)
            if counts[index]
        ],
        loc="upper left",
        fontsize=9,
    )
    axes[0].set_title(f"{counts[0]}/{len(outcome)} succeeded")

    command = data["command"][alive]
    axes[1].scatter(command[:, 1], command[:, 0], s=5, alpha=0.15, color="#1f5fc4")
    axes[1].set_xlabel("wz [rad/s]")
    axes[1].set_ylabel("vx [m/s]")
    axes[1].grid(alpha=0.25)
    axes[1].set_title(f"{len(command)} executed commands")

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
