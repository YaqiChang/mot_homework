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
from data_gen import Measurement, in_jam_region
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
    简化 LMB 扩展目标跟踪器（回到作业原始结构，在其基础上只做轻微改动）：
    - 候选轨迹由外部（main / dynamic_sim）显式添加（A、B），内部不再从杂波新生目标；
    - r 的上 / 下调节逻辑基本沿用原始版本，只额外区分“干扰区漏检”和“普通漏检”；
    - 这样可以保证：
        * B 在真值第一次出现时就能被加入并快速被检测到（r 初始较高）；
        * 0–200 s 内内部 ID 只分配一次，不会不断自增。
    该存在概率管理思想参考了标准多目标跟踪 / LMB 文献中常见的
    “存在概率 r + track management（多次连续命中 / 多次连续漏检）”做法。
    """

    def __init__(self):
        self.state = LMBState(components=[])
        self.next_label_id = 0
        self.R_meas = np.eye(2) * MEAS_STD_POS**2

    # ---------- Track 管理（由外部控制何时“加入候选轨迹”） ----------

    def _new_label(self) -> int:
        lab = self.next_label_id
        self.next_label_id += 1
        return lab

    def add_target_with_state(self, x0: np.ndarray, r0: float = 0.8) -> None:
        """
        用完整真值状态初始化一个扩展目标：
        x0 = [px, py, vx, vy]；
        r0 < 1，之后由量测慢慢把 r 拉到接近 1。
        """
        ext_state = ExtendedTargetState.init_from_state(x0)
        comp = BernoulliComponent(
            label=self._new_label(),
            r=r0,
            state=ext_state,
        )
        self.state.components.append(comp)

    def add_target_from_position(self, pos0: np.ndarray, r0: float = 0.7) -> None:
        """
        只有位置时的初始化（速度设为 0，由量测逐渐校正）。
        动态仿真里可以用量测质心做这种初始化。
        """
        ext_state = ExtendedTargetState.init_from_position(pos0)
        comp = BernoulliComponent(
            label=self._new_label(),
            r=r0,
            state=ext_state,
        )
        self.state.components.append(comp)

    # ---------- 滤波步骤 ----------

    def predict(self) -> None:
        """对所有 Bernoulli 做预测，并按生存概率轻微衰减 r。"""
        for comp in self.state.components:
            comp.state.predict()
            # 原始版本这里是关闭的，这里恢复一个温和的生存衰减
            comp.r *= SURVIVAL_PROB

    def _associate_measurements(
        self, meas_k: List[Measurement]
    ) -> (Dict[int, List[np.ndarray]], List[np.ndarray]):
        """简单 gating + 最近邻分配，返回：
        - assigned_measurements[i]: 分给第 i 个组件的量测列表；
        - unassigned_measurements: 没被任何轨迹解释的量测（目前不再用于“出生”）。
        """
        non_clutter = [m for m in meas_k if not m.is_clutter]
        assigned_measurements: Dict[int, List[np.ndarray]] = {
            i: [] for i in range(len(self.state.components))
        }
        unassigned_measurements: List[np.ndarray] = []

        if len(non_clutter) == 0 or len(self.state.components) == 0:
            for m in non_clutter:
                unassigned_measurements.append(m.z)
            return assigned_measurements, unassigned_measurements

        candidate_pairs = []  # (dist, mi, ci)
        for mi, m in enumerate(non_clutter):
            z = m.z
            dists = []
            for ci, comp in enumerate(self.state.components):
                pos_pred = comp.state.x[:2]
                dist = np.linalg.norm(z - pos_pred)
                if dist <= GATING_THRESHOLD:
                    dists.append((dist, ci))

            if len(dists) == 0:
                unassigned_measurements.append(z)
                continue

            dists.sort(key=lambda x: x[0])
            best_dist, best_ci = dists[0]
            second_dist = dists[1][0] if len(dists) > 1 else np.inf

            if second_dist - best_dist < ASSIGN_DIST_MARGIN:
                # 模糊量测，不分配给任何目标
                unassigned_measurements.append(z)
                continue

            candidate_pairs.append((best_dist, mi, best_ci))

        candidate_pairs.sort(key=lambda x: x[0])
        used_meas = set()
        for dist, mi, ci in candidate_pairs:
            if mi in used_meas:
                continue
            used_meas.add(mi)
            z = non_clutter[mi].z
            assigned_measurements[ci].append(z)

        for mi, m in enumerate(non_clutter):
            if mi not in used_meas:
                unassigned_measurements.append(m.z)

        return assigned_measurements, unassigned_measurements

    def update(self, meas_k: List[Measurement], t: float) -> None:
        """根据当前帧量测对所有 Bernoulli 做一次更新，并调整 r。"""
        assigned_measurements, _ = self._associate_measurements(meas_k)

        for idx, comp in enumerate(self.state.components):
            Z = (
                np.array(assigned_measurements[idx])
                if len(assigned_measurements[idx]) > 0
                else None
            )
            pos_pred = comp.state.x[:2]

            if Z is None:
                # 没有量测命中该目标：
                # - 在干扰区内：认为主要是被压制，只保留生存衰减；
                # - 其他情况：额外轻微衰减 r，促使长时间漏检的目标最终“死亡”。
                if in_jam_region(pos_pred, t):
                    # 干扰区：只用生存衰减，不再额外打压 r
                    pass
                else:
                    # 普通漏检：原始作业里用 0.9，这里保持不变
                    comp.r *= 0.9
            else:
                # 有量测支撑：根据点/扩展量测更新，并适度提升 r
                if Z.shape[0] == 1:
                    comp.state.update_with_point(Z[0], self.R_meas)
                else:
                    comp.state.update_with_extended(Z, self.R_meas)

                comp.r = min(0.99, comp.r + 0.1)

        self._prune_components()

    def _prune_components(self) -> None:
        """
        只删掉存在概率极低的组件，并限制最多 2 条轨迹：
        - 作业场景中只显式添加 A、B，不在内部重建新 label；
        - 因此这里只做“很小 r 的清理”和“最多保留 2 条轨迹”的保护。
        """
        self.state.components = [c for c in self.state.components if c.r > 0.01]
        if len(self.state.components) > 2:
            self.state.components.sort(key=lambda c: c.r, reverse=True)
            self.state.components = self.state.components[:2]

    # ---------- 上层接口 ----------

    def step(self, meas_k: List[Measurement], t: float) -> None:
        """单步递推：外部已经负责何时加入/删除候选轨迹，这里只做 r 的估计。"""
        self.predict()
        self.update(meas_k, t)

    def get_current_estimates(self) -> Dict[Any, np.ndarray]:
        """
        返回当前时刻各 track 的位置估计：
        - 只返回存在概率 r ≥ EXISTENCE_THRESHOLD 的组件；
        - Cardinality = 这些组件的个数。
        """
        est: Dict[Any, np.ndarray] = {}
        for comp in self.state.components:
            if comp.r >= EXISTENCE_THRESHOLD:
                est[comp.label] = comp.state.x[:2].copy()
        return est
