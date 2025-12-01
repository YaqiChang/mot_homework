# data_gen.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from config import (
    DT, F_POS, TOTAL_TIME, N_STEPS, KM,
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
    if t < JAM_START_T or t > JAM_END_T:
        return False
    dx = pos[0] - JAM_REGION_CENTER[0]
    dy = pos[1] - JAM_REGION_CENTER[1]
    # 中心 JAM_REGION_CENTER，正方形半边长 JAM_REGION_RADIUS
    return (abs(dx) <= JAM_REGION_RADIUS) and (abs(dy) <= JAM_REGION_RADIUS)

def generate_target_trajectories() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成两目标 A、B 的真值轨迹：
    - A：从 (60+25*√3km, 0) 向左匀速运动
    - B：在 t=20s 出现，与 A 在 t≈60s 附近交汇并机动
    """
    times = np.arange(0, TOTAL_TIME, DT)

    # 状态转移矩阵
    F = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    # ===== 新增：交汇点和交汇时间（离散步） =====
    F_pos = np.array([25.0 * KM, 0.0])   # F = (25 km, 0)
    meet_step = int(60.0 / DT)           # 约在 t ≈ 60 s 交汇（可调）
    # ---- 目标 A ----
    # A 从 1 号雷达正 x 方向 +60km 处出发：
    # R1.x = RADARS["R1"][0]，单位是 m
    xA0 = np.array([
        (25.0 * np.sqrt(3.0) + 60.0) * KM,
        0.0,
        -TARGET_SPEED,
        0.0
    ])
    
    trajA = np.zeros((N_STEPS, 4))
    trajA[0] = xA0
    for k in range(1, N_STEPS):
        trajA[k] = F @ trajA[k-1]
    
    # ===== A 到 F 的连续时间 & 离散步数 =====
    # A 是匀速：x_A(t) = xA0 - TARGET_SPEED * t
    # 要满足 x_A(t_meet) = 25km
    t_meet_cont = (xA0[0] - F_pos[0]) / TARGET_SPEED  # 单位：秒

    # 对应的离散步：
    meet_step = int(round(t_meet_cont / DT))
    meet_step = max(0, min(meet_step, N_STEPS - 1))   # 保证在范围内
    t_meet = times[meet_step]

    # ---- 目标 B ----
    appear_step_B = int(20.0 / DT)            # B 在 t=20s 出现
    trajB = np.full((N_STEPS, 4), np.nan)

    # 1) 交汇点 F 固定为 (25 km, 0)
    F_pos = np.array([25.0 * KM, 0.0])

    # 2) 根据 A 的速度，反算 A 到 F 的连续时间 & 对应的离散步
    #    A 是匀速：x_A(t) = xA0_x + v_A_x * t，v_A_x = trajA[0,2] (通常为 -TARGET_SPEED)
    vA = abs(trajA[0, 2])                     # A 的速度标量（m/s）
    t_meet_cont = (trajA[0, 0] - F_pos[0]) / vA   # 连续时间下 A 到 F 所需时间（秒）

    meet_step = int(round(t_meet_cont / DT))      # 对应的离散步
    meet_step = max(0, min(meet_step, N_STEPS - 1))
    t_meet = times[meet_step]

    # 3) B 的起点：B0 = (25√3 + 30, 12.5) km
    start_pos_B = np.array([
        (25.0 * np.sqrt(3.0) + 30.0) * KM,
        12.5 * KM
    ])

    # 从 B 出现（t = 20s）到交汇时刻 t_meet 的时间
    start_t_B = times[appear_step_B]
    duration_B = t_meet - start_t_B          # B 直线段可用时间

    if duration_B <= 0:
        duration_B = DT                      # 极端兜底，避免除 0

    # ---------- 段 1：y>0，B0 -> F，严格走直线 ----------
    # 这里直接做几何插值：B0 → F，是一条 y = kx + b 的斜线
    xB0 = start_pos_B.copy()
    vB = (F_pos - start_pos_B) / duration_B  # 平均速度 = 路程 / 时间（不会强制成 TARGET_SPEED）

    # 初始化 t=20s 时的状态
    xB = np.array([start_pos_B[0], start_pos_B[1], vB[0], vB[1]])
    trajB[appear_step_B] = xB

    for k in range(appear_step_B + 1, meet_step + 1):
        t_k = times[k]
        alpha = (t_k - start_t_B) / duration_B    # alpha 从 0 → 1

        # 几何上严格沿 B0-F 直线插值
        pos = (1.0 - alpha) * start_pos_B + alpha * F_pos

        # 速度用差分计算，不再归一化到 TARGET_SPEED
        vel = (pos - trajB[k - 1, :2]) / DT
        trajB[k, :2] = pos
        trajB[k, 2:] = vel

    # 强制确保交汇帧的坐标精确落在 F_pos（消除离散误差）
    trajB[meet_step, :2] = F_pos

    # 4) 段 2：y<0，从 F 向 3 号雷达附近弯下去（ln 形、向下凸）
    # 3 号雷达在 (0, -25km)，我们让 B 落在 (0, -20km) 附近即可
    dest = np.array([0.0, -20.0 * KM])

    remain_steps = N_STEPS - (meet_step + 1)
    if remain_steps > 0:
        prev_pos = F_pos.copy()
        prev_vel = trajB[meet_step, 2:].copy()

        beta = 4.0  # ln 曲率参数，越大弯曲越明显

        for i in range(remain_steps):
            k = meet_step + 1 + i
            s = (i + 1) / remain_steps     # s: 0 → 1

            # x：从 F.x 单调减小到 0（一路往 R3 的 x=0 靠近）
            x = (1.0 - s) * F_pos[0] + s * dest[0]

            # y：从 0 下降到 -20km，且整段“向下凸”
            # 设计一个 [0,1] 上的向下凸函数 y_norm(s)，然后整体乘 -20km
            # y_norm(s) = [ln(1+beta) - ln(1+beta*(1-s))] / ln(1+beta)
            y_norm = (np.log(1.0 + beta) - np.log(1.0 + beta * (1.0 - s))) / np.log(1.0 + beta)
            y = -20.0 * KM * y_norm

            pos = np.array([x, y])

            # 速度用差分计算，保持连续
            vel = (pos - prev_pos) / DT
            vnorm = np.linalg.norm(vel)
            if vnorm < 1e-3:
                vel = prev_vel

            trajB[k, :2] = pos
            trajB[k, 2:] = vel

            prev_pos = pos
            prev_vel = vel

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
