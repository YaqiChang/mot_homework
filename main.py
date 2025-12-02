# main.py
import numpy as np

from config import DT, N_STEPS
from data_gen import generate_target_trajectories, simulate_measurements, in_jam_region
from tracker_lmb import LMBTracker
from metrics import *

# 新增：可选绘图开关
ENABLE_PLOT = True

# 如果需要绘图，再导入 plot 模块
if ENABLE_PLOT:
    from plot import (
        plot_trajectories,
        plot_measurements_snapshot,
        plot_metrics_over_time,
        plot_id_timeline,
        plot_detection_recall,
        plot_confusion_matrix,
    )

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


    # --------- 新增：基数误差 + 检测率 ---------
    mean_card_err, card_bias = compute_cardinality_error(
        times=times,
        true_traj_dict=true_traj_dict,
        est_series=est_series,
    )
    print(f"Mean |cardinality error|: {mean_card_err:.2f}")
    print(f"Cardinality bias (est - true): {card_bias:.2f}")

    # 各目标的检测召回率（在存在的帧中，有多少帧被 tracker 成功“跟上”）
    maskA_exist = ~np.isnan(trajA[:, 0])
    maskB_exist = ~np.isnan(trajB[:, 0])
    detA = (~np.isnan(estA[:, 0])) & maskA_exist
    detB = (~np.isnan(estB[:, 0])) & maskB_exist
    recall_A = detA.sum() / max(maskA_exist.sum(), 1)
    recall_B = detB.sum() / max(maskB_exist.sum(), 1)
    print(f"Detection recall A: {recall_A:.2%}")
    print(f"Detection recall B: {recall_B:.2%}")
    
    
    # 计算整体多目标 OSPA 距离（集合层面）
    mean_ospa = compute_ospa_over_time(
        times=times,
        true_traj_dict=true_traj_dict,
        est_series=est_series,
        p=2,
        c=1000.0,   # 截断距离，根据场景尺度可调
    )
    print(f"Mean OSPA distance: {mean_ospa:.2f} m")

    # --------- 5. 混淆矩阵（ID vs 真值 A/B） ---------
    C = compute_confusion_matrix(
        trajA_true=trajA[:, :2],
        trajB_true=trajB[:, :2],
        estA=estA,
        estB=estB,
    )
    print("Confusion matrix (rows: True A/B; cols: Pred A/B/None):")
    print(C)

    # 由混淆矩阵派生的更直观统计量
    a_as_b = int(C[0, 1])
    b_as_a = int(C[1, 0])
    a_missed = int(C[0, 2])
    b_missed = int(C[1, 2])
    print(f"A 被误认为 B 的帧数: {a_as_b}")
    print(f"B 被误认为 A 的帧数: {b_as_a}")
    print(f"A 漏检帧数: {a_missed}")
    print(f"B 漏检帧数: {b_missed}")

    # --------- 6. 准备 ID–真值 时间轴信息（供绘图使用） ---------
    # 对每一帧构造 label -> truth_id 的映射；未在 label2truth 中的记为 "None"
    id_assign_series = []
    for k in range(N_STEPS):
        mapping = {}
        est = est_series[k]
        for lab in est.keys():
            tid = label2truth.get(lab, "None")
            mapping[lab] = tid
        id_assign_series.append(mapping)

    # --------- 7. 准备随时间变化的召回率曲线 ---------
    cum_A_exist = np.cumsum(maskA_exist.astype(int))
    cum_B_exist = np.cumsum(maskB_exist.astype(int))
    cum_A_det = np.cumsum(detA.astype(int))
    cum_B_det = np.cumsum(detB.astype(int))

    recall_A_t = np.zeros_like(times, dtype=float)
    recall_B_t = np.zeros_like(times, dtype=float)
    for k in range(N_STEPS):
        recall_A_t[k] = (cum_A_det[k] / cum_A_exist[k]) if cum_A_exist[k] > 0 else 0.0
        recall_B_t[k] = (cum_B_det[k] / cum_B_exist[k]) if cum_B_exist[k] > 0 else 0.0

    recalls_dict = {"A": recall_A_t, "B": recall_B_t}

    # --------- 8. 统计假目标总数及干扰区内的假警次数 ---------
    # 使用逐帧最近邻分配（build_id_assignment_series）来判定哪些 track 为“未匹配到真值”
    nn_assign_series = build_id_assignment_series(
        times=times,
        true_traj_dict=true_traj_dict,
        est_series=est_series,
    )

    false_total = 0
    false_in_jam = 0
    for k, t in enumerate(times):
        est_k = est_series[k]
        assign_k = nn_assign_series[k]
        for lab, pos in est_k.items():
            tid = assign_k.get(lab, "None")
            if tid == "None":
                false_total += 1
                if in_jam_region(pos, t):
                    false_in_jam += 1

    print(f"假目标（未匹配任何真值）的总次数: {false_total}")
    print(f"其中位于干扰区内的假警次数: {false_in_jam}")

    # --------- 9. 保存一份指标摘要到文本文件 ---------
    with open("results/metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("Tracking metrics summary\n")
        f.write("------------------------\n")
        f.write(f"RMSE A (position): {rmse_A:.4f} m\n")
        f.write(f"RMSE B (position): {rmse_B:.4f} m\n")
        f.write(f"Mean |cardinality error|: {mean_card_err:.4f}\n")
        f.write(f"Cardinality bias (est - true): {card_bias:.4f}\n")
        f.write(f"Detection recall A: {recall_A:.4%}\n")
        f.write(f"Detection recall B: {recall_B:.4%}\n")
        f.write(f"Mean OSPA distance: {mean_ospa:.4f} m\n")
        f.write("Confusion matrix (rows: True A/B; cols: Pred A/B/None):\n")
        f.write(f"{C[0, :]}\n")
        f.write(f"{C[1, :]}\n")
        f.write(f"A_as_B: {a_as_b}, B_as_A: {b_as_a}\n")
        f.write(f"A_missed: {a_missed}, B_missed: {b_missed}\n")
        f.write(f"False targets total (unmatched tracks): {false_total}\n")
        f.write(f"False targets inside jamming area: {false_in_jam}\n")

    if ENABLE_PLOT:
        plot_trajectories(
            times=times,
            trajA_true=trajA[:, :2],
            trajB_true=trajB[:, :2],
            estA=estA,
            estB=estB,
            save_path="results/traj_result.png",
            show=False,
        )

        # OSPA / RMSE / Cardinality 随时间变化的曲线
        plot_metrics_over_time(
            times=times,
            trajA_true=trajA[:, :2],
            trajB_true=trajB[:, :2],
            est_series=est_series,
            estA=estA,
            estB=estB,
            p=2,
            c=1000.0,
            save_path="results/metrics_over_time.png",
            show=False,
        )

        # ID–真值 时间轴图
        plot_id_timeline(
            times=times,
            assign_series=id_assign_series,
            save_path="results/id_timeline.png",
            show=False,
        )

        # Recall 随时间变化曲线
        plot_detection_recall(
            times=times,
            recalls=recalls_dict,
            save_path="results/detection_recall.png",
            show=False,
        )

        # 混淆矩阵热力图
        plot_confusion_matrix(
            C=C,
            save_path="results/confusion_matrix.png",
            show=False,
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
                save_path="results/meas_step60.png",
                show=False
            )


if __name__ == "__main__":
    main()
