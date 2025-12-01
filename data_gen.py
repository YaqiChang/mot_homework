# data_gen.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from config import (
    DT, TOTAL_TIME, N_STEPS, KM,
    RADARS, POINT_EXT_THRESHOLD,
    JAM_REGION_CENTER, JAM_REGION_RADIUS,
    JAM_START_T, JAM_END_T,
    CLUTTER_REGION_X, CLUTTER_REGION_Y, CLUTTER_INTENSITY,
    STATE_STD_POS, STATE_STD_VEL,
    MEAS_STD_POS, TARGET_SPEED
)

@dataclass
class TargetState:
    """单目标状态向量封装"""
    x: np.ndarray  # shape (4,) [px, py, vx, vy]

@dataclass
class Measurement:
    """单个量测点"""
    z: np.ndarray       # (px, py)
    radar_id: str
    is_clutter: bool = False
    true_target_id: str = None  # A/B 用于验证和了解数据

def in_jam_region(pos: np.ndarray, t: float) -> bool:
    """判断目标在 t 时刻是否处于对 R2 的遮蔽/压制干扰区"""
    if t < JAM_START_T or t > JAM_END_T:
        return False
    return np.linalg.norm(pos - JAM_REGION_CENTER) <= JAM_REGION_RADIUS

def generate_target_trajectories() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成两目标 A、B 的真值轨迹：
    - A：从 (60km, 0) 向左匀速运动
    - B：在 t=20s 出现，与 A 在 t≈60s 附近交汇并机动
    """
    times = np.arange(0, TOTAL_TIME, DT)

    # 状态转移矩阵
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # ---- 目标 A ----
    xA0 = np.array([60 * KM, 0.0, -TARGET_SPEED, 0.0])
    trajA = np.zeros((N_STEPS, 4))
    trajA[0] = xA0
    for k in range(1, N_STEPS):
        trajA[k] = F @ trajA[k-1]

    # ---- 目标 B ----
    appear_step_B = int(20.0 / DT)
    trajB = np.full((N_STEPS, 4), np.nan)

    meet_step = int(60.0 / DT)
    meet_pos = trajA[meet_step, :2]

    start_t_B = times[appear_step_B]
    end_t_meet = times[meet_step]
    duration = end_t_meet - start_t_B

    start_pos_B = meet_pos + np.array([20 * KM, -20 * KM])
    vB = (meet_pos - start_pos_B) / duration
    vB_norm = np.linalg.norm(vB)
    if vB_norm > 1e-3:
        vB = vB * (TARGET_SPEED / vB_norm)

    xB = np.array([start_pos_B[0], start_pos_B[1], vB[0], vB[1]])
    trajB[appear_step_B] = xB

    for k in range(appear_step_B + 1, N_STEPS):
        xB = F @ xB
        trajB[k] = xB

    # 交汇后机动（改变 B 的速度方向）
    maneuver_start_step = meet_step
    maneuver_angle = np.deg2rad(30)
    R_rot = np.array([[np.cos(maneuver_angle), -np.sin(maneuver_angle)],
                      [np.sin(maneuver_angle),  np.cos(maneuver_angle)]])

    for k in range(maneuver_start_step, N_STEPS):
        if np.isnan(trajB[k, 0]):
            continue
        if k == maneuver_start_step:
            vel = trajB[k, 2:]
            new_vel = R_rot @ vel
            new_vel = new_vel * (TARGET_SPEED / np.linalg.norm(new_vel))
            trajB[k, 2:] = new_vel
        if k > maneuver_start_step:
            trajB[k] = F @ trajB[k-1]

    return times, trajA, trajB

# --------- 扩展量测 / 点量测 / 杂波 --------------

def generate_extended_measurements(pos_true: np.ndarray, n_scatter: int = 10) -> np.ndarray:
    """
    简单扩展目标模型：在目标真值附近采样散射点 + 噪声。
    返回 shape (n_scatter, 2)。
    """
    extent_cov = np.array([[100**2, 0],
                           [0, 50**2]])  # 100m x 50m 尺度
    scatters = np.random.multivariate_normal(pos_true, extent_cov, size=n_scatter)
    noise = np.random.normal(0, MEAS_STD_POS, size=(n_scatter, 2))
    return scatters + noise

def generate_point_measurement(pos_true: np.ndarray) -> np.ndarray:
    """点目标量测 = 真值 + 高斯噪声"""
    noise = np.random.normal(0, MEAS_STD_POS, size=2)
    return pos_true + noise

def generate_clutter_points() -> List[np.ndarray]:
    """按面积和强度生成均匀杂波点"""
    area = (CLUTTER_REGION_X[1] - CLUTTER_REGION_X[0]) * (CLUTTER_REGION_Y[1] - CLUTTER_REGION_Y[0])
    lam = CLUTTER_INTENSITY * area
    k = np.random.poisson(lam)
    if k <= 0:
        return []
    xs = np.random.uniform(CLUTTER_REGION_X[0], CLUTTER_REGION_X[1], size=k)
    ys = np.random.uniform(CLUTTER_REGION_Y[0], CLUTTER_REGION_Y[1], size=k)
    return [np.array([x, y]) for x, y in zip(xs, ys)]

def simulate_measurements(times, trajA, trajB) -> List[List[Measurement]]:
    """
    对每个时刻、每部雷达，生成：
    - 目标 A/B 的点/扩展量测
    - 对应遮蔽 / 干扰
    - 杂波
    返回：all_measurements[time_step] = [Measurement, ...]
    """
    all_measurements: List[List[Measurement]] = []

    for step, t in enumerate(times):
        meas_k: List[Measurement] = []

        # 为每个雷达生成杂波
        radar_clutter: Dict[str, List[np.ndarray]] = {
            r_id: generate_clutter_points() for r_id in RADARS.keys()
        }

        for radar_id, radar_pos in RADARS.items():
            # ---- 目标 A ----
            posA = trajA[step, :2]
            rA = np.linalg.norm(posA - radar_pos)

            # R2 遮蔽
            if not (radar_id == "R2" and in_jam_region(posA, t)):
                if rA < POINT_EXT_THRESHOLD:
                    zs = generate_extended_measurements(posA)
                    for z in zs:
                        meas_k.append(Measurement(z=z, radar_id=radar_id, is_clutter=False, true_target_id="A"))
                else:
                    z = generate_point_measurement(posA)
                    meas_k.append(Measurement(z=z, radar_id=radar_id, is_clutter=False, true_target_id="A"))

            # ---- 目标 B ----
            if not np.isnan(trajB[step, 0]):
                posB = trajB[step, :2]
                rB = np.linalg.norm(posB - radar_pos)
                if not (radar_id == "R2" and in_jam_region(posB, t)):
                    if rB < POINT_EXT_THRESHOLD:
                        zs = generate_extended_measurements(posB)
                        for z in zs:
                            meas_k.append(Measurement(z=z, radar_id=radar_id, is_clutter=False, true_target_id="B"))
                    else:
                        z = generate_point_measurement(posB)
                        meas_k.append(Measurement(z=z, radar_id=radar_id, is_clutter=False, true_target_id="B"))

            # ---- 杂波 ----
            for zc in radar_clutter[radar_id]:
                meas_k.append(Measurement(z=zc, radar_id=radar_id, is_clutter=True, true_target_id=None))

        all_measurements.append(meas_k)

    return all_measurements
