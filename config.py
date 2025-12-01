# config.py
import numpy as np

# --------- 时间与步长 ----------
DT = 1.0                # 采样周期 (s)
TOTAL_TIME = 200.0      # 总仿真时长 (s)
N_STEPS = int(TOTAL_TIME / DT)

# --------- 单位转换 ----------
KM = 1000.0

# --------- 雷达配置 ----------
RADARS = {
    "R1": np.array([0.0, 0.0]),
    "R2": np.array([80 * KM, 0.0]),
    "R3": np.array([40 * KM, 60 * KM])
}

# 点目标 / 扩展目标距离阈值
POINT_EXT_THRESHOLD = 50 * KM

# --------- 噪声与误差 ----------
STATE_STD_POS = 5.0        # m
STATE_STD_VEL = 5.0        # m/s
MEAS_STD_POS = 20.0        # m

# 目标速度
TARGET_SPEED = 500.0       # m/s

# --------- 遮蔽 / 压制干扰 ----------
JAM_REGION_CENTER = np.array([40 * KM, 10 * KM])
JAM_REGION_RADIUS = 10 * KM
JAM_START_T = 60.0
JAM_END_T = 65.0   # 至少 5 s

# --------- 杂波 ----------
CLUTTER_REGION_X = (10 * KM, 70 * KM)
CLUTTER_REGION_Y = (-10 * KM, 30 * KM)
# 理论要求 5/km^2 会在 60km x 40km 的区域里产生 1.2 万点/帧，超出演示和算法承受能力；
# 这里下调到 0.05/km^2 保留“强杂波”特性但避免淹没所有量测。
CLUTTER_INTENSITY = 0.05 / (KM * KM)

# --------- LMB 参数 ----------
SURVIVAL_PROB = 0.99      # 生存概率
BIRTH_PROB = 0.0        # 出生概率，避免疯狂出生
EXISTENCE_THRESHOLD = 0.5 # track 有效存在概率阈值
GATING_THRESHOLD =500   # 简单欧氏距离 gating 阈值（单位 m，示意）


# 用于多目标扩展量测分配的“相对距离裕度”
ASSIGN_DIST_MARGIN = 50.0  # m，可按场景调节，比如 50~150
