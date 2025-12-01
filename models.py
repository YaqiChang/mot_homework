# models.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import (
    DT, STATE_STD_POS, STATE_STD_VEL
)

STATE_DIM = 4  # [px, py, vx, vy]

# 基本状态转移矩阵（简单 CV）
F = np.array([[1, 0, DT, 0],
              [0, 1, 0, DT],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

Q = np.diag([1.0, 1.0, 10.0, 10.0])  # 过程噪声（可以调整）

@dataclass
class ExtendedTargetState:
    """
    扩展目标状态：
    - x: [px, py, vx, vy]
    - P: 4x4 协方差
    - S: 2x2 扩展形状矩阵
    - gamma: 期望回波点数
    """
    def __init__(self, x, P, S, gamma=10.0):
        self.x = x          # numpy array (4,)
        self.P = P          # numpy array (4,4)
        self.S = S          # numpy array (2,2)
        self.gamma = gamma  # float

    @staticmethod
    def init_from_position(pos: np.ndarray):
        """
        仅有位置时的初始化，速度默认 0。
        """
        x0 = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        P0 = np.diag([
            STATE_STD_POS**2, STATE_STD_POS**2,
            STATE_STD_VEL**2, STATE_STD_VEL**2
        ])
        S0 = np.array([[100**2, 0],
                       [0, 50**2]], dtype=float)
        return ExtendedTargetState(x=x0, P=P0, S=S0, gamma=10.0)

    @staticmethod
    def init_from_state(x0: np.ndarray):
        """
        正确的“用真值初始化”接口：
        x0 = [px, py, vx, vy]
        """
        assert x0.shape[0] == 4
        P0 = np.diag([
            STATE_STD_POS**2, STATE_STD_POS**2,
            (5*STATE_STD_VEL)**2, (5*STATE_STD_VEL)**2
        ])
        S0 = np.array([[100**2, 0],
                       [0, 50**2]], dtype=float)
        return ExtendedTargetState(x=x0.copy(), P=P0, S=S0, gamma=10.0)

    def predict(self):
        """CV 模型预测（S 与 gamma 暂保持不变）"""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_with_point(self, z: np.ndarray, R: np.ndarray):
        """单点量测更新"""
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        y = z - H @ self.x
        S_cov = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S_cov)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_with_extended(self, Z: np.ndarray, R: np.ndarray):
        """
        扩展量测更新：
        - 用点集均值更新位置
        - 用点集协方差更新形状
        """
        if Z.shape[0] == 0:
            return

        z_mean = Z.mean(axis=0)
        z_cov = np.cov(Z.T) if Z.shape[0] > 1 else np.eye(2) * 100.0

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R_eff = z_cov / max(1, Z.shape[0]) + R

        y = z_mean - H @ self.x
        S_cov = H @ self.P @ H.T + R_eff
        K = self.P @ H.T @ np.linalg.inv(S_cov)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        # 扩展形状指数平滑
        alpha = 0.1
        self.S = (1 - alpha) * self.S + alpha * z_cov
        self.gamma = 0.9 * self.gamma + 0.1 * float(Z.shape[0])

@dataclass
class BernoulliComponent:
    """
    LMB 中的一个 Bernoulli 组件：
    - label: 唯一标签（比如整数/字符串）
    - r: 存在概率
    - state: 扩展目标状态
    """
    label: Any
    r: float
    state: ExtendedTargetState

@dataclass
class LMBState:
    """LMB 状态：若干 Bernoulli 组件的集合"""
    components: List[BernoulliComponent] = field(default_factory=list)
