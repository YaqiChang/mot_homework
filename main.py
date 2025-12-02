# main.py
import numpy as np

from config import DT, N_STEPS
from data_gen import generate_target_trajectories, simulate_measurements
from tracker_lmb import LMBTracker
from metrics import compute_rmse, align_labels_to_truth, compute_ospa_over_time

# 新增：可选绘图开关
ENABLE_PLOT = True

# 如果需要绘图，再导入 plot 模块（避免无 matplotlib 时也能跑算法）
if ENABLE_PLOT:
    from plot import plot_trajectories, plot_measurements_snapshot

def main():
    times, trajA, trajB = generate_target_trajectories()
    measurements = simulate_measurements(times, trajA, trajB)

    tracker = LMBTracker()

    # --------- 1. 初始化 A：从 t=0 开始，用真值状态 ----------
    xA0 = trajA[0, :4]          # [px, py, vx, vy]
    tracker.add_target_with_state(xA0)

    # --------- 2. 找到 B 的首次出现时刻 ----------
    first_B_idx = np.where(~np.isnan(trajB[:, 0]))[0][0]
    xB0 = trajB[first_B_idx, :4]    # [px, py, vx, vy] at first_B_idx

    est_series = []

    # --------- 3. 主循环 ----------
    for k in range(N_STEPS):
        t = k * DT
        meas_k = measurements[k]

        if k == first_B_idx:
            # 在 B 第一次出现的时刻，把 B 加入 tracker
            tracker.add_target_with_state(xB0)

            # 这一帧先不做 predict/update，直接记录当前估计（A + 新出生的 B）
            est = tracker.get_current_estimates()
            est_series.append(est)
            continue   # 跳过本帧的 step，下一帧再开始正常 predict+update

        # 其他时刻：正常 predict + update
        tracker.step(meas_k, t)
        est = tracker.get_current_estimates()
        est_series.append(est)

    # --------- 4. 对齐 label -> A/B，计算 RMSE（保持你原来的写法即可） ---------
    true_traj_dict = {
        "A": trajA[:, :2],
        "B": trajB[:, :2]
    }
    label2truth = align_labels_to_truth(times, true_traj_dict, est_series)
    print("Label 对真值目标的匹配关系:", label2truth)

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

    rmse_A = compute_rmse(trajA[:, :2], estA)
    rmse_B = compute_rmse(trajB[:, :2], estB)
    print(f"RMSE A (position): {rmse_A:.2f} m")
    print(f"RMSE B (position): {rmse_B:.2f} m")


    # 计算整体多目标 OSPA 距离（集合层面）
    mean_ospa = compute_ospa_over_time(
        times=times,
        true_traj_dict=true_traj_dict,
        est_series=est_series,
        p=2,
        c=1000.0   # 截断距离，根据场景尺度可调
    )
    print(f"Mean OSPA distance: {mean_ospa:.2f} m")

    if ENABLE_PLOT:
        plot_trajectories(
            times=times,
            trajA_true=trajA[:, :2],
            trajB_true=trajB[:, :2],
            estA=estA,
            estB=estB,
            save_path="traj_result.png",
            show=False
        )

        snapshot_idx = 60
        if 0 <= snapshot_idx < len(measurements):
            trueA_k = trajA[snapshot_idx, :2]
            trueB_k = trajB[snapshot_idx, :2]
            estA_k = estA[snapshot_idx] if not np.isnan(estA[snapshot_idx, 0]) else None
            estB_k = estB[snapshot_idx] if not np.isnan(estB[snapshot_idx, 0]) else None

            plot_measurements_snapshot(
                measurements_k=measurements[snapshot_idx],
                step_idx=snapshot_idx,
                trueA=trueA_k,
                trueB=trueB_k,
                estA=estA_k,
                estB=estB_k,
                save_path="meas_step60.png",
                show=False
            )


if __name__ == "__main__":
    main()
