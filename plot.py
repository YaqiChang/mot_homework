# plot.py
import matplotlib
matplotlib.use("Agg")  # 避免 Qt xcb 问题，只出图片文件
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Optional, Dict

from data_gen import Measurement
from metrics import _ospa_single_timestep


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
    ax.plot(trajB_true[:, 0], trajB_true[:, 1], color='purple', linestyle='-', label="True B")

    # 估计（A/B 用点线，区别于真值实线）
    if estA is not None:
        maskA = ~np.isnan(estA[:, 0])
        ax.plot(
            estA[maskA, 0],
            estA[maskA, 1],
            linestyle=':',
            color='orange',
            label="Est A",
        )
    if estB is not None:
        maskB = ~np.isnan(estB[:, 0])
        ax.plot(
            estB[maskB, 0],
            estB[maskB, 1],
            linestyle=':',
            color='red',
            label="Est B",
        )

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


def plot_metrics_over_time(
    times: np.ndarray,
    trajA_true: np.ndarray,
    trajB_true: np.ndarray,
    est_series: List[Dict],
    estA: np.ndarray,
    estB: np.ndarray,
    p: int = 2,
    c: float = 1000.0,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    绘制随时间变化的 OSPA / RMSE / Cardinality 曲线，
    逻辑与 dynamic_sim 右侧指标面板保持一致。
    """
    T = len(times)

    ospa_vals = np.zeros(T, dtype=float)
    rmse_vals = np.zeros(T, dtype=float)
    card_true = np.zeros(T, dtype=float)
    card_est = np.zeros(T, dtype=float)

    for k in range(T):
        # 真实集合 X_k
        X_list = []
        if not np.isnan(trajA_true[k, 0]):
            X_list.append(trajA_true[k])
        if not np.isnan(trajB_true[k, 0]):
            X_list.append(trajB_true[k])
        X = np.array(X_list, dtype=float) if len(X_list) > 0 else np.zeros((0, 2))

        # 估计集合 Y_k（不做 label 对齐，集合层面）
        est_k = est_series[k]
        Y_list = [pos for _, pos in est_k.items()]
        Y = np.array(Y_list, dtype=float) if len(Y_list) > 0 else np.zeros((0, 2))

        ospa_vals[k] = _ospa_single_timestep(X, Y, p=p, c=c)

        # 即时 RMSE（与 dynamic_sim 中的定义保持一致）
        sq_errs = []
        if (not np.isnan(trajA_true[k, 0])) and (not np.isnan(estA[k, 0])):
            diffA = trajA_true[k] - estA[k]
            sq_errs.append(np.sum(diffA ** 2))
        if (not np.isnan(trajB_true[k, 0])) and (not np.isnan(estB[k, 0])):
            diffB = trajB_true[k] - estB[k]
            sq_errs.append(np.sum(diffB ** 2))
        rmse_vals[k] = np.sqrt(np.mean(sq_errs)) if sq_errs else 0.0

        # Cardinality
        ct = 0
        if not np.isnan(trajA_true[k, 0]):
            ct += 1
        if not np.isnan(trajB_true[k, 0]):
            ct += 1
        card_true[k] = ct
        card_est[k] = float(len(est_k))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax_ospa, ax_rmse, ax_card = axes

    ax_ospa.plot(times, ospa_vals, linewidth=1.8)
    ax_ospa.set_ylabel("OSPA [m]")
    ax_ospa.set_title("OSPA Distance Over Time")
    ax_ospa.grid(True, alpha=0.3)

    ax_rmse.plot(times, rmse_vals, linewidth=1.8)
    ax_rmse.set_ylabel("RMSE [m]")
    ax_rmse.set_title("Position RMSE Over Time")
    ax_rmse.grid(True, alpha=0.3)

    ax_card.plot(times, card_est, linewidth=1.8, label="Est")
    ax_card.plot(times, card_true, "k--", linewidth=1.2, label="Truth")
    ax_card.set_ylabel("Cardinality")
    ax_card.set_xlabel("Time [s]")
    ax_card.set_title("Target Cardinality Over Time")
    ax_card.grid(True, alpha=0.3)
    ax_card.legend(fontsize=9)

    fig.tight_layout()

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

    # 叠加真值/估计（如果给），颜色与主轨迹图保持一致：
    # True A: 蓝色，True B: 紫色，Est A: 橙色，Est B: 红色
    if trueA is not None and not np.isnan(trueA[0]):
        ax.plot(trueA[0], trueA[1], marker='x', color='blue', markersize=10, linestyle='None', label="True A")
    if trueB is not None and not np.isnan(trueB[0]):
        ax.plot(trueB[0], trueB[1], marker='x', color='purple', markersize=10, linestyle='None', label="True B")
    if estA is not None and not np.isnan(estA[0]):
        ax.plot(estA[0], estA[1], marker='+', color='orange', markersize=10, linestyle='None', label="Est A")
    if estB is not None and not np.isnan(estB[0]):
        ax.plot(estB[0], estB[1], marker='+', color='red', markersize=10, linestyle='None', label="Est B")

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


def plot_metrics_time_series(
    times: np.ndarray,
    ospa_vals: np.ndarray,
    rmse_vals: np.ndarray,
    card_true: np.ndarray,
    card_est: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    画 OSPA / RMSE / Cardinality 三个时序指标子图。
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax_ospa, ax_rmse, ax_card = axes

    ax_ospa.plot(times, ospa_vals, linewidth=1.5)
    ax_ospa.set_ylabel("OSPA")
    ax_ospa.grid(True)

    ax_rmse.plot(times, rmse_vals, linewidth=1.5)
    ax_rmse.set_ylabel("RMSE [m]")
    ax_rmse.grid(True)

    ax_card.plot(times, card_true, "k--", linewidth=1.2, label="Truth")
    ax_card.plot(times, card_est, linewidth=1.5, label="Est")
    ax_card.set_ylabel("Cardinality")
    ax_card.set_xlabel("Time [s]")
    ax_card.legend()
    ax_card.grid(True)

    fig.suptitle("GLMB Performance over Time", fontsize=14)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    C: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    绘制 2x3 混淆矩阵热力图：
        行: true A/B
        列: pred A / pred B / pred None
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))

    im = ax.imshow(C, cmap="Blues", vmin=0)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Pred A", "Pred B", "Pred None"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True A", "True B"])

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(
                j,
                i,
                int(C[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title("ID Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_id_timeline(
    times: np.ndarray,
    assign_series: List[Dict[Any, str]],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    画 ID–时间 关系图：
        Y 轴：filter 中的 label；
        不同颜色：所属真值目标（A/B/None）。
    """
    # 收集所有 label
    labels = set()
    for d in assign_series:
        labels |= set(d.keys())
    labels = sorted(
        labels,
        key=lambda x: int(x) if isinstance(x, (int, np.integer)) else str(x),
    )

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_labels = len(labels)
    T = len(times)

    # 为不同 truth_id 分配颜色
    colors = {"A": "tab:blue", "B": "tab:orange", "None": "tab:gray"}

    fig, ax = plt.subplots(figsize=(8, 4))

    for k in range(T):
        mapping = assign_series[k]
        for lab, tid in mapping.items():
            y = label_to_idx[lab]
            c = colors.get(tid, "tab:gray")
            ax.scatter(times[k], y, s=10, c=c)

    ax.set_yticks(range(n_labels))
    ax.set_yticklabels([str(lab) for lab in labels])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Label ID")
    ax.set_title("Label–Truth Assignment over Time")

    # 图例
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors["A"], label="A"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors["B"], label="B"),
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=colors["None"],
            label="None/Unmatched"
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_detection_recall(
    times: np.ndarray,
    recalls: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    画 A/B 的累积召回率曲线。
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for tid, arr in recalls.items():
        ax.plot(times, arr, label=f"Recall {tid}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()
    ax.set_title("Detection Recall over Time")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
