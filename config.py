# config.py
import numpy as np

# --------- 时间与步长 ----------
DT = 1.0                # 采样周期 (s)
TOTAL_TIME = 200.0      # 总仿真时长 (s)
N_STEPS = int(TOTAL_TIME / DT)

# --------- 单位转换 ----------
KM = 1000.0

# --------- 雷达配置 ----------

# 三部雷达构成边长 50km 的等边三角形
RADARS = {
    # 1 号雷达在 x 轴上：x = 25 * sqrt(3) km
    "R1": np.array([25.0 * np.sqrt(3.0) * KM, 0.0]),      # ≈ (43.3km, 0)

    # 2、3 号雷达在 y 轴上，对称
    "R2": np.array([0.0,  25.0 * KM]),                    # (0, 25km)
    "R3": np.array([0.0, -25.0 * KM]),                    # (0,-25km)
}

# 25 * sqrt(3) km ≈ 43.3 km
# F 点在 x 轴上，位置大约是 (2/3)*25*sqrt(3) km
F_X = 25 * KM   # ≈ 25 km
F_POS = np.array([F_X, 0.0])

# 点目标 / 扩展目标距离阈值
POINT_EXT_THRESHOLD = 50 * KM

# --------- 噪声与误差 ----------
STATE_STD_POS = 5.0        # m
STATE_STD_VEL = 5.0        # m/s
MEAS_STD_POS = 20.0        # m

# 目标速度
TARGET_SPEED = 500.0       # m/s

# --------- 遮蔽 / 压制干扰 ----------
# 交汇点 F 位置（在 x 轴上，和 A 轨迹重合）
F_POS = np.array([25.0 * KM, 0.0])

# 遮蔽干扰：在 F 点右侧约 5km 的正方形区域
# R1.x = 25√3 km，F_X = (2/3)*25√3 km
R1_X = RADARS["R1"][0]

# --- 干扰区域：位于 F 和 R1 中点 ---
# 中心在 (F.x + R1.x) / 2, y=0
JAM_REGION_CENTER = np.array([(F_POS[0] + R1_X) / 2.0, 0.0])


# 半边长 2.5km → 正方形边长约 5km（完全在 F 和 R1 之间）
JAM_REGION_RADIUS = 2.5 * KM

# 时间你可以保留原来的
JAM_START_T = 40.0    # 举例
JAM_END_T   = 80.0


# --------- 杂波 ----------
CLUTTER_REGION_X = (-40 * KM, 110 * KM)  # 覆盖 A 的起点 (≈103km)
CLUTTER_REGION_Y = (-60 * KM, 60 * KM)
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
