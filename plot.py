# plot.py
import matplotlib
matplotlib.use("Agg")  # 避免 Qt xcb 问题，只出图片文件
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict

from data_gen import Measurement


def plot_trajectories(
    times: np.ndarray,
    trajA_true: np.ndarray,
    trajB_true: np.ndarray,
    estA: np.ndarray,
    estB: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """真值 + 估计轨迹总览"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 真值
    ax.plot(trajA_true[:, 0], trajA_true[:, 1], 'b-', label="True A")
    ax.plot(trajB_true[:, 0], trajB_true[:, 1], 'r-', label="True B")

    # 估计
    if estA is not None:
        maskA = ~np.isnan(estA[:, 0])
        ax.plot(estA[maskA, 0], estA[maskA, 1], 'b--', label="Est A")
    if estB is not None:
        maskB = ~np.isnan(estB[:, 0])
        ax.plot(estB[maskB, 0], estB[maskB, 1], 'r--', label="Est B")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("True vs Estimated Trajectories")
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_measurements_snapshot(
    measurements_k: List[Measurement],
    step_idx: int,
    trueA: Optional[np.ndarray] = None,
    trueB: Optional[np.ndarray] = None,
    estA: Optional[np.ndarray] = None,
    estB: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    画某一帧的量测点云：
    - 按 radar_id 区分颜色
    - 区分 clutter vs 非 clutter
    - 可选叠加该帧的真值位置/估计位置
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 按雷达分组
    radar_groups: Dict[str, List[Measurement]] = {}
    for m in measurements_k:
        radar_groups.setdefault(m.radar_id, []).append(m)

    # 为不同雷达准备不同 marker
    markers = ['o', 's', '^', 'x', 'd', 'v', 'P', '*']
    radar_ids = list(radar_groups.keys())

    for i, rid in enumerate(radar_ids):
        group = radar_groups[rid]
        pts = np.array([m.z for m in group])
        is_clutter = np.array([m.is_clutter for m in group])

        if pts.shape[0] == 0:
            continue

        # 非杂波
        if np.any(~is_clutter):
            ax.scatter(
                pts[~is_clutter, 0],
                pts[~is_clutter, 1],
                marker=markers[i % len(markers)],
                alpha=0.7,
                label=f"Radar {rid} (target)",
            )
        # 杂波
        if np.any(is_clutter):
            ax.scatter(
                pts[is_clutter, 0],
                pts[is_clutter, 1],
                marker=markers[i % len(markers)],
                alpha=0.3,
                edgecolors='none',
                label=f"Radar {rid} (clutter)",
            )

    # 叠加真值/估计（如果给）
    if trueA is not None and not np.isnan(trueA[0]):
        ax.plot(trueA[0], trueA[1], 'bx', markersize=10, label="True A")
    if trueB is not None and not np.isnan(trueB[0]):
        ax.plot(trueB[0], trueB[1], 'rx', markersize=10, label="True B")
    if estA is not None and not np.isnan(estA[0]):
        ax.plot(estA[0], estA[1], 'b+', markersize=10, label="Est A")
    if estB is not None and not np.isnan(estB[0]):
        ax.plot(estB[0], estB[1], 'r+', markersize=10, label="Est B")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Measurements at step {step_idx}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
