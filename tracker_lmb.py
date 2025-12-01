# tracker_lmb.py
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from models import ExtendedTargetState, BernoulliComponent, LMBState
from config import (
    GATING_THRESHOLD, RADARS, MEAS_STD_POS, STATE_STD_POS, STATE_STD_VEL,
    SURVIVAL_PROB, BIRTH_PROB,
    EXISTENCE_THRESHOLD, POINT_EXT_THRESHOLD
)
from data_gen import Measurement
from models import (
    ExtendedTargetState, BernoulliComponent, LMBState
)
from config import MEAS_STD_POS, GATING_THRESHOLD, STATE_STD_POS, STATE_STD_VEL, ASSIGN_DIST_MARGIN

from utils import simple_gating, cluster_extended_measurements

@dataclass
class TrackEstimate:
    """输出给上层的轨迹估计"""
    label: Any
    positions: List[np.ndarray]   # 每步 [px, py]

class LMBTracker:
    """
    简化 LMB 扩展目标跟踪器：
    - 不做完整的多假设，只做“每个 Bernoulli 对应一个轨迹”；
    - 数据关联用简单 gating + 最近邻；
    - 重点是结构清晰，方便课程作业说明。
    """
    def __init__(self):
        self.state = LMBState(components=[])
        self.next_label_id = 0
        # 观测噪声矩阵
        self.R_meas = np.eye(2) * MEAS_STD_POS**2
        
    def _new_label(self) -> int:
        lab = self.next_label_id
        self.next_label_id += 1
        return lab
    
    def add_target(self, pos0: np.ndarray):
            """
            在当前时刻添加一个新目标（比如 A 在 t=0，B 在出现时）。
            用pos0作为初始位置，速度设为0（由量测逐步修正）。
            """
            ext_state = ExtendedTargetState.init_from_position(pos0)
            comp = BernoulliComponent(
                label=self._new_label(),
                r=0.9,
                state=ext_state
            )
            self.state.components.append(comp)
            
    def add_target_with_state(self, x0: np.ndarray):
        """
        用完整真值状态初始化扩展目标：
        x0 = [px, py, vx, vy]
        """
        # 这里直接用你给的 ExtendedTargetState 构造函数
        P0 = np.diag([STATE_STD_POS**2, STATE_STD_POS**2,
                      (5 * STATE_STD_VEL)**2, (5 * STATE_STD_VEL)**2])
        S0 = np.array([[100**2, 0],
                       [0, 50**2]], dtype=float)
        ext_state = ExtendedTargetState(x=x0.copy(), P=P0, S=S0, gamma=10.0)

        comp = BernoulliComponent(
            label=self._new_label(),
            r=0.9,
            state=ext_state
        )
        self.state.components.append(comp)
                
    def initialize_from_truth(self, posA0: np.ndarray, posB0: np.ndarray):
        """
        用真值初始位置初始化两个扩展目标 A、B。
        - posA0: 目标 A 初始位置 (2,)
        - posB0: 目标 B 初始出现时的位置 (2,)
        """
        self.state.components = []
        self.next_label_id = 0

        extA = ExtendedTargetState.init_from_position(posA0)
        compA = BernoulliComponent(label=self._new_label(), r=0.9, state=extA)

        extB = ExtendedTargetState.init_from_position(posB0)
        compB = BernoulliComponent(label=self._new_label(), r=0.9, state=extB)

        self.state.components = [compA, compB]
        self.initialized = True
        
    def init_two_targets_from_measurements(self, meas_k):
        """
        已知场景中只有两个目标（A,B），
        从第0帧量测中粗略初始化两个扩展目标 track。
        简化起见：对所有非杂波点做聚类，取前两个簇作为 A/B。
        """
        from utils import cluster_extended_measurements
        from data_gen import Measurement

        non_clutter = [m for m in meas_k if not m.is_clutter]
        # 聚类半径可以稍微小一点，比如 200m
        clusters = cluster_extended_measurements(non_clutter, distance_thresh=200.0)

        if len(clusters) == 0:
            return

        # 按簇大小排序，取前两个（如果只有一个就只建一个）
        clusters.sort(key=lambda c: c.shape[0], reverse=True)
        num_tracks = min(2, len(clusters))

        for i in range(num_tracks):
            pos_init = clusters[i].mean(axis=0)
            ext_state = ExtendedTargetState.init_from_position(pos_init)
            comp = BernoulliComponent(
                label=self._new_label(),
                r=0.9,
                state=ext_state
            )
            self.state.components.append(comp)

        self.initialized = True
    # ---------- Track 管理 ----------

    # def initialize_from_first_frame(self, meas_k: List[Measurement]):
    #     """从第一帧量测初始化若干 track（简单：对非杂波量测聚类，每簇一个出生 Bernoulli）"""
    #     non_clutter = [m for m in meas_k if not m.is_clutter]
    #     clusters = cluster_extended_measurements(non_clutter, distance_thresh=200.0)

    #     for cl in clusters:
    #         pos_init = cl.mean(axis=0)
    #         ext_state = ExtendedTargetState.init_from_position(pos_init)
    #         comp = BernoulliComponent(
    #             label=self._new_label(),
    #             r=0.9,
    #             state=ext_state
    #         )
    #         self.state.components.append(comp)

    def _new_label(self) -> int:
        lab = self.next_label_id
        self.next_label_id += 1
        return lab

    # ---------- 滤波步骤 ----------

    def predict(self):
        """对所有 Bernoulli 做预测，并衰减存在概率"""
        for comp in self.state.components:
            comp.state.predict()
            comp.r *= SURVIVAL_PROB

    def update(self, meas_k: List[Measurement]):
        """
        简化更新：
        1. 收集所有非杂波量测；
        2. 在 gate 内构造所有 (轨迹, 量测) 配对，按距离全局排序，贪心分配；
           ——保证每个量测至多只被一个轨迹使用，避免 A 抢光所有量测；
        3. 对每个 Bernoulli 根据点/扩展量测更新状态；
        4. 未分配量测目前只记入 unassigned（题目中目标数已知，不再出生新目标）。
        """
        # 1) 过滤掉杂波量测（is_clutter = True 的不参与分配）
        non_clutter = [m for m in meas_k if not m.is_clutter]

        # 为每个组件准备一个量测集合（扩展/点混合）
        assigned_measurements: Dict[int, List[np.ndarray]] = {
            i: [] for i in range(len(self.state.components))
        }
        unassigned_measurements: List[np.ndarray] = []

        # 2) 构造“轨迹-量测”候选配对，并全局贪心分配
        if len(non_clutter) > 0 and len(self.state.components) > 0:
            candidate_pairs = []  # (dist, mi, ci)

            for mi, m in enumerate(non_clutter):
                z = m.z
                # 计算该点到所有目标预测中心的距离
                dists = []
                for ci, comp in enumerate(self.state.components):
                    pos_pred = comp.state.x[:2]
                    dist = np.linalg.norm(z - pos_pred)
                    if dist <= GATING_THRESHOLD:
                        dists.append((dist, ci))

                if len(dists) == 0:
                    # 对所有目标都不在 gate 内，视为未分配 / 杂波
                    unassigned_measurements.append(z)
                    continue

                # 按距离排序，找最近和次近
                dists.sort(key=lambda x: x[0])
                best_dist, best_ci = dists[0]
                if len(dists) > 1:
                    second_dist = dists[1][0]
                else:
                    second_dist = np.inf

                # 如果这个点对最近目标的“优势”不够明显（距离差小于裕度），
                # 认为是模糊点，不分配给任何目标，避免 A/B 抢同一团
                if second_dist - best_dist < ASSIGN_DIST_MARGIN:
                    unassigned_measurements.append(z)
                    continue

                # 否则才把这个点作为 (mi, best_ci) 的候选
                candidate_pairs.append((best_dist, mi, best_ci))

            # 然后再做一次全局贪心（其实现在每个点最多只出现一次）
            candidate_pairs.sort(key=lambda x: x[0])
            used_meas = set()

            for dist, mi, ci in candidate_pairs:
                if mi in used_meas:
                    continue  # 这个量测已经被分配给别的轨迹
                used_meas.add(mi)
                z = non_clutter[mi].z
                assigned_measurements[ci].append(z)

            # 把那些在某些 gate 内但最终没被用的量测，也视作未分配（可选）
            for mi, m in enumerate(non_clutter):
                if mi not in used_meas:
                    unassigned_measurements.append(m.z)
        else:
            # 没有非杂波量测或没有轨迹，所有量测都视为未分配
            for m in non_clutter:
                unassigned_measurements.append(m.z)

        # 3) 对每个组件做更新
        for idx, comp in enumerate(self.state.components):
            Z = np.array(assigned_measurements[idx]) if len(assigned_measurements[idx]) > 0 else None
            if Z is None:
                # 没有量测分配，存在概率略衰减
                comp.r *= 0.9
            else:
                # 判断是点量测还是扩展量测：根据点数量粗判
                if Z.shape[0] == 1:
                    comp.state.update_with_point(Z[0], self.R_meas)
                else:
                    comp.state.update_with_extended(Z, self.R_meas)
                # 有量测支撑，存在概率向上拉
                comp.r = min(0.99, comp.r + 0.1)

        # 4) 不再对未分配量测触发新生（目标数已知为 2）
        # for z in unassigned_measurements:
        #     ...

        # 5) 清理不存在的组件（存在概率过低）
        self._prune_components()

    def _prune_components(self):
    # 只删掉存在概率极低的组件，避免误删 A/B
        self.state.components = [
            c for c in self.state.components if c.r > 0.01
        ]
        # 如果仍然超过2个（极端情况下），保留存在概率最高的两个
        if len(self.state.components) > 2:
            self.state.components.sort(key=lambda c: c.r, reverse=True)
            self.state.components = self.state.components[:2]

    # ---------- 上层接口 ----------

    def step(self, meas_k):
        """
        单步：对已有目标做预测+更新。
        （注意：A/B 的添加由外部 main 控制，不在这里自动出生）
        """
        # 如果当前还没有任何目标（极端情况），直接返回
        if len(self.state.components) == 0:
            return

        self.predict()
        self.update(meas_k)

    def get_current_estimates(self) -> Dict[Any, np.ndarray]:
        """返回当前时刻各 track 的位置估计（只返回存在概率足够高的）"""
        est = {}
        for comp in self.state.components:
            if comp.r >= EXISTENCE_THRESHOLD:
                est[comp.label] = comp.state.x[:2].copy()
        return est
