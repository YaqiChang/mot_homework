# dynamic_sim.py
"""
动态场景仿真与可视化（可选调用）
- 读取 config / data_gen / tracker_lmb
- 做一步一步的动画，可选择保存 GIF
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle
import numpy as np

from config import (
    KM, RADARS,
    JAM_REGION_CENTER, JAM_REGION_RADIUS,
    TOTAL_TIME, N_STEPS,
    EXISTENCE_THRESHOLD,
)
from data_gen import generate_target_trajectories, simulate_measurements, in_jam_region
from tracker_lmb import LMBTracker
from metrics import _ospa_single_timestep

# 使用 TkAgg，防止某些环境下卡死（失败就忽略）
try:
    matplotlib.use("TkAgg")
except Exception:
    pass


def get_ellipse_params(pos, S, n_std: float = 2.0):
    """
    根据位置和形状矩阵 S 计算椭圆参数 (center, width, height, angle)
    S: 2x2 形状矩阵（类似协方差）
    """
    vals, vecs = np.linalg.eigh(S)

    # 大特征值对应长轴
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # 方向角（弧度 -> 度）
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 轴长：2 * n_std * sqrt(λ)
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-3))

    return pos, width, height, theta


def main(save_gif: bool = True, gif_name: str = "result.gif"):
    """
    动态场景仿真入口
    :param save_gif: 是否保存 GIF 动画
    :param gif_name: GIF 文件名
    """
    # 1. 真值 + 量测
    print("[dynamic_sim] 生成真值轨迹...")
    times, trajA, trajB = generate_target_trajectories()

    print("[dynamic_sim] 生成量测数据...")
    all_measurements = simulate_measurements(times, trajA, trajB)

    # 2. LMB 跟踪器
    tracker = LMBTracker()

    # 3. 绘图框架
    fig = plt.figure(figsize=(18, 9), facecolor='#2b2b2b')
    gs = gridspec.GridSpec(4, 3, figure=fig)

    # --- 左侧：场景 ---
    ax_sim = fig.add_subplot(gs[:, :2])
    ax_sim.set_facecolor('black')
    ax_sim.set_title('Extended Target Tracking Simulation (LMB)', color='white', fontsize=16)
    ax_sim.set_xlabel('X (m)', color='white')
    ax_sim.set_ylabel('Y (m)', color='white')
    ax_sim.tick_params(colors='white')
    ax_sim.grid(True, linestyle='--', alpha=0.3)

    # 显示范围：覆盖 3 部雷达 + A 起点 + 交汇区
    ax_sim.set_xlim(-40 * KM, 110 * KM)
    ax_sim.set_ylim(-60 * KM, 60 * KM)

    # 雷达
    for rid, pos in RADARS.items():
        ax_sim.plot(pos[0], pos[1], 'r^', markersize=12, label='Radar' if rid == 'R1' else "")
        circle = Circle(pos, 60 * KM, color='red', fill=False, linestyle=':', alpha=0.3)
        ax_sim.add_artist(circle)
        ax_sim.text(pos[0], pos[1] - 3 * KM, rid, color='red', fontsize=10, ha='center')

    # 干扰区（针对 R2）
    jam_circle = Circle(JAM_REGION_CENTER, JAM_REGION_RADIUS, color='yellow', alpha=0.15)
    ax_sim.add_patch(jam_circle)
    ax_sim.text(JAM_REGION_CENTER[0], JAM_REGION_CENTER[1], 'Jamming\nArea',
                color='yellow', fontsize=9, ha='center', va='center')

    # 动态元素
    scat_meas, = ax_sim.plot([], [], 'g.', markersize=2, alpha=0.6, label='Measurements')

    # 真值轨迹线
    truth_lines = {
        'A': ax_sim.plot([], [], 'c--', linewidth=1, alpha=0.5)[0],
        'B': ax_sim.plot([], [], 'm--', linewidth=1, alpha=0.5)[0]
    }

    # 估计椭圆/标签缓存
    est_patches = []
    est_labels = []

    ax_sim.legend(loc='upper right', facecolor='#333333', edgecolor='white', labelcolor='white')

    # --- 右侧：指标 ---
    ax_ospa = fig.add_subplot(gs[0, 2], facecolor='#1f1f1f')
    ax_rmse = fig.add_subplot(gs[1, 2], facecolor='#1f1f1f')
    ax_card = fig.add_subplot(gs[2, 2], facecolor='#1f1f1f')
    ax_info = fig.add_subplot(gs[3, 2], facecolor='#1f1f1f')

    metrics_axes = [ax_ospa, ax_rmse, ax_card]
    titles = ['OSPA Distance', 'Position RMSE (m)', 'Cardinality (Target Num)']
    lines = []

    history = {
        'time': [],
        'ospa': [],
        'rmse': [],
        'card_true': [],
        'card_est': [],
    }

    for ax, title in zip(metrics_axes, titles):
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.2)
        line, = ax.plot([], [], 'y-', linewidth=1.5)
        lines.append(line)

    # Cardinality 真值线
    line_card_true, = ax_card.plot([], [], 'w--', linewidth=1, alpha=0.5, label='Truth')
    ax_card.legend(fontsize=8, facecolor='#333333', labelcolor='white')

    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.5, "Initializing...",
                             color='white', fontsize=10, va='center')

    def update(frame: int):
        t = times[frame]
        meas_k = all_measurements[frame]

        # 当前真值
        posA = trajA[frame, :2]
        posB = trajB[frame, :2]

        current_truth_states = {}
        if not np.isnan(trajA[frame, 0]):
            current_truth_states['A'] = posA
        if not np.isnan(trajB[frame, 0]):
            current_truth_states['B'] = posB

        # --- 跟踪器更新 ---
        if frame == 0:
            # 用第一帧量测初始化两个目标
            tracker.init_two_targets_from_measurements(meas_k)

        # 中途出现的 B（如果没 track 到，则强行加一个 Bernoulli）
        if frame > 10 and len(tracker.state.components) < 2:
            if 19 <= t <= 25 and 'B' in current_truth_states:
                is_tracked = False
                for c in tracker.state.components:
                    if np.linalg.norm(c.state.x[:2] - posB) < 3000.0:
                        is_tracked = True
                        break
                if not is_tracked:
                    tracker.add_target(posB + np.random.randn(2) * 50.0)

        # 一步 LMB 递推
        if hasattr(tracker, 'step'):
            tracker.step(meas_k)
        else:
            tracker.predict()
            tracker.update(meas_k)
            tracker._prune_components()

        # --- 提取估计 ---
        estimates = []
        est_pos_map = {}

        for comp in tracker.state.components:
            if comp.r >= EXISTENCE_THRESHOLD:
                estimates.append((comp.state.x[:2], comp.state.S, comp.label))
                est_pos_map[comp.label] = comp.state.x[:2]

        # --- 指标 ---
        X_true = (np.array(list(current_truth_states.values()))
                  if current_truth_states else np.zeros((0, 2)))
        Y_est = (np.array(list(est_pos_map.values()))
                 if est_pos_map else np.zeros((0, 2)))

        ospa_val = _ospa_single_timestep(X_true, Y_est, p=2, c=1000.0)

        sq_errs = []
        if 'A' in current_truth_states and len(est_pos_map) > 0:
            dists = [np.linalg.norm(p - current_truth_states['A'])
                     for p in est_pos_map.values()]
            if min(dists) < 2000.0:
                sq_errs.append(min(dists) ** 2)
        if 'B' in current_truth_states and len(est_pos_map) > 0:
            dists = [np.linalg.norm(p - current_truth_states['B'])
                     for p in est_pos_map.values()]
            if min(dists) < 2000.0:
                sq_errs.append(min(dists) ** 2)

        rmse_val = np.sqrt(np.mean(sq_errs)) if sq_errs else 0.0

        history['time'].append(t)
        history['ospa'].append(ospa_val)
        history['rmse'].append(rmse_val)
        history['card_true'].append(len(current_truth_states))
        history['card_est'].append(len(est_pos_map))

        # --- A. 量测点 ---
        zs = [m.z for m in meas_k]
        if len(zs) > 0:
            zs = np.array(zs)
            scat_meas.set_data(zs[:, 0], zs[:, 1])
        else:
            scat_meas.set_data([], [])

        # --- B. 真值轨迹 ---
        curr_traj_A = trajA[:frame + 1]
        valid_A = ~np.isnan(curr_traj_A[:, 0])
        truth_lines['A'].set_data(curr_traj_A[valid_A, 0], curr_traj_A[valid_A, 1])

        curr_traj_B = trajB[:frame + 1]
        valid_B = ~np.isnan(curr_traj_B[:, 0])
        truth_lines['B'].set_data(curr_traj_B[valid_B, 0], curr_traj_B[valid_B, 1])

        # --- C. 椭圆估计 ---
        for p in est_patches:
            p.remove()
        for l in est_labels:
            l.remove()
        est_patches.clear()
        est_labels.clear()

        for pos, S, label in estimates:
            e_pos, e_w, e_h, e_ang = get_ellipse_params(pos, S)
            ell = Ellipse(
                xy=e_pos,
                width=e_w,
                height=e_h,
                angle=e_ang,
                edgecolor='yellow',
                facecolor='none',
                linewidth=2,
            )
            ax_sim.add_patch(ell)
            est_patches.append(ell)

            txt = ax_sim.text(
                pos[0], pos[1] + 2000.0,
                f"ID:{label}",
                color='yellow',
                fontsize=9,
                fontweight='bold',
            )
            est_labels.append(txt)

        # --- D. 指标曲线 ---
        times_arr = history['time']
        lines[0].set_data(times_arr, history['ospa'])
        lines[1].set_data(times_arr, history['rmse'])
        lines[2].set_data(times_arr, history['card_est'])
        line_card_true.set_data(times_arr, history['card_true'])

        for ax in metrics_axes:
            ax.set_xlim(0, TOTAL_TIME)

        if history['ospa']:
            ax_ospa.set_ylim(0, max(max(history['ospa']), 150.0))
        if history['rmse']:
            ax_rmse.set_ylim(0, max(max(history['rmse']), 50.0))
        ax_card.set_ylim(0, 4)

        # --- E. 状态文本 ---
        status_str = f"Time: {t:.1f}s\n"
        status_str += f"Targets: {len(current_truth_states)} (True) / {len(est_pos_map)} (Est)\n"
        if 'A' in current_truth_states and in_jam_region(posA, t):
            status_str += "!! Target A in Jamming !!\n"
        info_text.set_text(status_str)

        return [scat_meas] + list(truth_lines.values()) + est_patches + lines

    ani = FuncAnimation(fig, update, frames=range(N_STEPS), interval=50, blit=False)

    if save_gif:
        print("[dynamic_sim] 正在生成 GIF，请稍等...")
        ani.save(gif_name, writer='pillow', fps=10)
        print(f"[dynamic_sim] GIF 已保存为 {gif_name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 单独运行 dynamic_sim.py 时，直接看动画
    main(save_gif=False)
