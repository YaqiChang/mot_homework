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
    TOTAL_TIME, N_STEPS, DT,
    EXISTENCE_THRESHOLD,
)
from data_gen import generate_target_trajectories, simulate_measurements, in_jam_region
from tracker_lmb import LMBTracker
from metrics import _ospa_single_timestep, align_labels_to_truth

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

    # 2. 先按与 main.py 完全一致的方式跑一遍滤波，记录每一帧的估计结果
    tracker = LMBTracker()

    # 与 main.py 一致：预先确定 B 的首次出现时刻，并缓存 A/B 的初始状态
    xA0 = trajA[0, :4]
    first_B_idx_arr = np.where(~np.isnan(trajB[:, 0]))[0]
    first_B_idx = int(first_B_idx_arr[0]) if len(first_B_idx_arr) > 0 else None
    xB0 = trajB[first_B_idx, :4] if first_B_idx is not None else None

    # 初始化 A
    tracker.add_target_with_state(xA0)

    est_series = []       # 与 main.py 中的 est_series 结构相同：每帧 {label: pos}
    ellipse_series = []   # 每帧 [(pos, S, label), ...]，用于后续画椭圆

    for k in range(N_STEPS):
        t = k * DT
        meas_k = all_measurements[k]

        if (first_B_idx is not None) and (k == first_B_idx):
            # B 首次出现：只把 B 加入 tracker，不做 step，与 main.py 保持一致
            tracker.add_target_with_state(xB0)
            est_simple = tracker.get_current_estimates()
        else:
            tracker.step(meas_k, t)
            est_simple = tracker.get_current_estimates()

        est_series.append({lab: pos.copy() for lab, pos in est_simple.items()})

        ell_list = []
        for comp in tracker.state.components:
            lab = comp.label
            if lab in est_simple:
                pos = est_simple[lab]
                ell_list.append((pos.copy(), comp.state.S.copy(), lab))
        ellipse_series.append(ell_list)

    # 与 main.py 相同方式进行 label -> A/B 的离线对齐
    true_traj_dict = {
        "A": trajA[:, :2],
        "B": trajB[:, :2],
    }
    label2truth = align_labels_to_truth(times, true_traj_dict, est_series)

    estA = np.full_like(trajA[:, :2], np.nan)
    estB = np.full_like(trajB[:, :2], np.nan)
    for k in range(N_STEPS):
        est = est_series[k]
        for lab, pos in est.items():
            if lab not in label2truth:
                continue
            tid = label2truth[lab]
            if tid == "A":
                estA[k] = pos
            elif tid == "B":
                estB[k] = pos

    # 预计算累计召回率，用于 info 面板显示（逻辑与 main.py 一致）
    A_exist_arr = ~np.isnan(trajA[:, 0])
    B_exist_arr = ~np.isnan(trajB[:, 0])
    A_det_arr = A_exist_arr & (~np.isnan(estA[:, 0]))
    B_det_arr = B_exist_arr & (~np.isnan(estB[:, 0]))

    cum_A_exist = np.cumsum(A_exist_arr.astype(int))
    cum_B_exist = np.cumsum(B_exist_arr.astype(int))
    cum_A_det = np.cumsum(A_det_arr.astype(int))
    cum_B_det = np.cumsum(B_det_arr.astype(int))

    # 3. 绘图框架
    # 稍微加大画布，便于在动图中看清各类元素
    fig = plt.figure(figsize=(20, 10), facecolor="#ffffff")  # 底色
    gs = gridspec.GridSpec(4, 3, figure=fig)

    # --- 左侧：场景 ---
    ax_sim = fig.add_subplot(gs[:, :2])
    ax_sim.set_facecolor('black')
    ax_sim.set_title('Extended Global Label Target Tracking Simulation (GLMB)',
                     color='white', fontsize=20, fontweight='bold')
    ax_sim.set_xlabel('X (m)', color='white', fontsize=14)
    ax_sim.set_ylabel('Y (m)', color='white', fontsize=14)
    ax_sim.tick_params(colors='white', labelsize=12)
    ax_sim.grid(True, linestyle='--', alpha=0.3)

    # 显示范围：覆盖 3 部雷达 + A 起点 + 交汇区
    ax_sim.set_xlim(-40 * KM, 110 * KM)
    ax_sim.set_ylim(-60 * KM, 60 * KM)

    # 雷达（形状与标注整体放大）
    for rid, pos in RADARS.items():
        ax_sim.plot(
            pos[0], pos[1], 'r^',
            markersize=10,
            label='Radar' if rid == 'R1' else ""
        )
        circle = Circle(
            pos, 60 * KM,
            edgecolor='red',
            facecolor='none',
            linestyle=':',
            linewidth=2.0,
            alpha=0.5,
        )
        ax_sim.add_artist(circle)
        ax_sim.text(
            pos[0], pos[1] - 3 * KM, rid,
            color='red',
            fontsize=14,
            fontweight='bold',
            ha='center',
        )

    # 干扰区（针对 R2）：更明亮的实心黄色区域 + 黑色加粗文字
    jam_circle = Circle(
        JAM_REGION_CENTER,
        JAM_REGION_RADIUS,
        edgecolor='yellow',
        facecolor='yellow',
        linewidth=2.0,
        alpha=0.4,
        label='Jamming Area',
    )
    ax_sim.add_patch(jam_circle)
    ax_sim.text(
        JAM_REGION_CENTER[0],
        JAM_REGION_CENTER[1],
        'Jamming\nArea',
        color='black',
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='center',
    )

    # 动态元素
    scat_meas, = ax_sim.plot([], [], 'g.', markersize=3.5, alpha=0.7, label='Measurements')

    # 真值轨迹线（B 画得更粗、更亮，便于观察）
    truth_lines = {
        'A': ax_sim.plot([], [], 'c--', linewidth=2.0, alpha=0.8)[0],
        'B': ax_sim.plot([], [], linestyle='--', color='orange', linewidth=3.0, alpha=0.9)[0],
    }

    # 当前真值点（大号 marker，便于观察当前帧 A/B 位置）
    point_markers = {
        'A': ax_sim.plot([], [], 'co', markersize=12, label='A true')[0],
        'B': ax_sim.plot([], [], marker='o', color='orange', markersize=12, label='B true')[0],
    }

    # 估计椭圆/标签缓存
    est_patches = []
    est_labels = []

    # 估计轨迹线：A/B 用与 main.py 对齐后的轨迹（线条略加粗）
    est_lines = {
        'A': ax_sim.plot([], [], linestyle='-', color='blue', linewidth=2.0, alpha=0.9, label='A est')[0],
        'B': ax_sim.plot([], [], 'm-', linewidth=2.0, alpha=0.9, label='B est')[0],
    }

    ax_sim.legend(
        loc='upper right',
        facecolor='white',
        edgecolor='black',
        labelcolor='black',
        fontsize=12,
    )

    # --- 右侧：指标 ---
    ax_ospa = fig.add_subplot(gs[0, 2], facecolor='white')
    ax_rmse = fig.add_subplot(gs[1, 2], facecolor='white')
    ax_card = fig.add_subplot(gs[2, 2], facecolor='white')
    ax_info = fig.add_subplot(gs[3, 2], facecolor='white')

    metrics_axes = [ax_ospa, ax_rmse, ax_card]
    titles = ['OSPA Distance', 'Position RMSE (m)', 'Cardinality (Target Num)']
    lines = []

    history = {
        'time': [],
        'ospa': [],
        'rmse': [],
        'card_true': [],
        'card_est': [],
        'card_err': [],   # 估计目标数 - 真实目标数，用来在线计算 |card err| 和 bias
    }
    
    for ax, title in zip(metrics_axes, titles):
        ax.set_title(title, color='black', fontsize=12, fontweight='bold')
        ax.tick_params(colors='black', labelsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        # 为 Cardinality 轴单独加上估计曲线的图例
        if ax is ax_card:
            line, = ax.plot([], [], linewidth=2.0, label='Est')
        else:
            line, = ax.plot([], [], linewidth=2.0)
        lines.append(line)

    # Cardinality 真值线
    line_card_true, = ax_card.plot([], [], 'k--', linewidth=1.5, alpha=0.8, label='Truth')
    ax_card.legend(fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')

    ax_info.axis('off')
    info_text = ax_info.text(
        0.05, 0.5, "Initializing...",
        color='black',
        fontsize=12,
        va='center',
    )

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

        # --- 提取估计（使用预先计算好的 est_series / ellipse_series，与 main.py 对齐） ---
        est_simple = est_series[frame]  # Dict[label, pos]
        est_pos_map = dict(est_simple)
        estimates = ellipse_series[frame]

        # --- 指标 ---
        X_true = (np.array(list(current_truth_states.values()))
                  if current_truth_states else np.zeros((0, 2)))
        Y_est = (np.array(list(est_pos_map.values()))
                 if est_pos_map else np.zeros((0, 2)))

        ospa_val = _ospa_single_timestep(X_true, Y_est, p=2, c=1000.0)

        # 即时 RMSE（按 main.py 中 estA/estB 的对齐结果）
        sq_errs = []
        det_A = False
        det_B = False

        if (not np.isnan(trajA[frame, 0])) and (not np.isnan(estA[frame, 0])):
            diffA = trajA[frame, :2] - estA[frame]
            sq_errs.append(np.sum(diffA ** 2))
            det_A = True

        if (not np.isnan(trajB[frame, 0])) and (not np.isnan(estB[frame, 0])):
            diffB = trajB[frame, :2] - estB[frame]
            sq_errs.append(np.sum(diffB ** 2))
            det_B = True

        rmse_val = np.sqrt(np.mean(sq_errs)) if sq_errs else 0.0

        history['time'].append(t)
        history['ospa'].append(ospa_val)
        history['rmse'].append(rmse_val)
        history['card_true'].append(len(current_truth_states))
        history['card_est'].append(len(est_pos_map))
        history['card_err'].append(len(est_pos_map) - len(current_truth_states))

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

        # 当前真值点，大号 marker 单独标出来
        if not np.isnan(trajA[frame, 0]):
            point_markers['A'].set_data(posA[0], posA[1])
        else:
            point_markers['A'].set_data([], [])

        if not np.isnan(trajB[frame, 0]):
            point_markers['B'].set_data(posB[0], posB[1])
        else:
            point_markers['B'].set_data([], [])

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
                edgecolor='red',
                facecolor='none',
                linewidth=2.5,
            )
            ax_sim.add_patch(ell)
            est_patches.append(ell)

            truth_id = label2truth.get(label, "?")
            txt = ax_sim.text(
                pos[0], pos[1] + 2000.0,
                f"{truth_id}(ID{label})",
                color='red',
                fontsize=11,
                fontweight='bold',
            )
            est_labels.append(txt)

        # --- D. 估计轨迹线（按与 main.py 相同的对齐结果画 A/B） ---
        for tid, traj_est, line in (
            ('A', estA, est_lines['A']),
            ('B', estB, est_lines['B']),
        ):
            curr = traj_est[:frame + 1]
            valid = ~np.isnan(curr[:, 0])
            line.set_data(curr[valid, 0], curr[valid, 1])

        # --- E. 指标曲线 ---
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

        # --- F. 状态文本（加入在线召回率 & 基数误差统计） ---
        status_str = f"Time: {t:.1f}s\n"
        status_str += f"Targets: {len(current_truth_states)} (True) / {len(est_pos_map)} (Est)\n"

        # A/B 在线召回率（按 main.py 对齐后的 estA/estB）
        if cum_A_exist[frame] > 0:
            recA = cum_A_det[frame] / cum_A_exist[frame]
        else:
            recA = 0.0
        if cum_B_exist[frame] > 0:
            recB = cum_B_det[frame] / cum_B_exist[frame]
        else:
            recB = 0.0

        # 基数误差：到当前时刻的平均 |err| 和 bias
        mean_abs_ce = np.mean(np.abs(history['card_err']))
        bias_ce = np.mean(history['card_err'])

        status_str += f"Recall A/B: {recA:.1%} / {recB:.1%}\n"
        status_str += f"|CardErr| mean: {mean_abs_ce:.2f}, bias: {bias_ce:.2f}\n"

        if 'A' in current_truth_states and in_jam_region(posA, t):
            status_str += "!! Target A in Jamming !!\n"

        info_text.set_text(status_str)


        return (
            [scat_meas]
            + list(truth_lines.values())
            + list(point_markers.values())
            + est_patches
            + lines
        )

    ani = FuncAnimation(fig, update, frames=range(N_STEPS), interval=50, blit=False)

    if save_gif:
        print("[dynamic_sim] 正在生成 GIF，请稍等...")
        ani.save(gif_name, writer='pillow', fps=10)
        print(f"[dynamic_sim] GIF 已保存为 {gif_name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os
    # 每轮开始前清屏，上一轮的输出全部清掉
    os.system("clear")      # Windows 用 "cls"
    main()
