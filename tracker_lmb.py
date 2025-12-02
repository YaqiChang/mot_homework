# tracker_lmb.py
# === GLMB 数学说明 ===
# 本文件实现的是一个针对 2 目标场景的简化 δ-GLMB：
# - 状态模型:  x_k = F x_{k-1} + q_k
# - 量测模型:  z_k ~ N(H x_k, R)，每个目标每帧至多产生 1 个量测
# - 多目标后验: π(X) = Δ(X) * ∑_c w^(c)(L(X)) [ p^(c) ]^X
#   我们用 GLMBHypothesis 表示 c，用 BernoulliComponent 表示单目标项。
# - 更新时，将每个量测看作：
#       * 被某条 track 检测到 (概率 P_D)
#       * 或者是杂波 (概率 1 - P_D)
#   并在假设权重 w^(c) 里累乘对应似然。
#
# 参考:
# [1] Vo B.N. et al., "An efficient implementation of the GLMB filter", TSP 2016.
# [2] Do C.T. et al., "Multi-object tracking with an adaptive GLMB filter", Sig.Proc. 2022.
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


@dataclass
class GLMBHypothesis:
    """
    简化 GLMB 假设：
    - weight: 该全局关联历史 c 的权重；
    - components: 在该假设下，各 label 的 Bernoulli 组件（r 和状态）。
    """
    weight: float
    components: List[BernoulliComponent]


class LMBTracker:
    """
    简化版 GLMB/LMB 跟踪器（针对本作业场景做的剪裁版实现）：
    - 内部维护若干条“全局假设”（GLMB 分量），每条假设都有自己的 A/B 轨迹状态和 r；
    - 每帧对模糊量测（同时落入多个轨迹 gate 内）保留多种关联方案，而不是贪心丢弃；
    - 对外仍然以“单个 LMB 近似”提供接口：取权重最高的假设作为当前状态；
    - r 的管理基本沿用原始作业版本，只在干扰区单独处理漏检。

    该结构实现的是一个非常简化的 GLMB 思路：
    - 维护少量假设 c ∈ C；
    - 每个假设有自己的 {Bernoulli 组件}；
    - 每帧根据量测生成新的假设集合并剪枝。
    """

    def __init__(self):
        # GLMB 假设集合
        self.hypotheses: List[GLMBHypothesis] = []

        # 供外部访问的“当前 LMB 状态”（取自权重最高的假设）
        self.state = LMBState(components=[])

        self.next_label_id = 0
        self.R_meas = np.eye(2) * MEAS_STD_POS**2

        # 每帧最多保留的 GLMB 假设数（防止组合爆炸）
        # 略微增加，避免 B 在模糊量测时过早被剪枝掉
        self.max_hypotheses = 8
        
        # === 新增：检测 / 杂波 先验参数（GLMB 的关键） ===
        # P_D: 目标被检测到的概率；这里对所有轨迹先统一用一个常数
        # 略微提高 P_D，鼓励“把量测解释为目标”而不是杂波
        self.P_D = 0.95
        # P_FA: 当前量测是“杂波”的先验概率（简化模型下与 P_D 对偶）
        self.P_FA = 1.0 - self.P_D

    # ---------- Track 管理（由外部控制何时“加入候选轨迹”） ----------

    def _new_label(self) -> int:
        lab = self.next_label_id
        self.next_label_id += 1
        return lab

    def _clone_component(self, comp: BernoulliComponent) -> BernoulliComponent:
        """深拷贝一个 BernoulliComponent，避免不同假设之间状态相互污染。"""
        s = comp.state
        new_state = ExtendedTargetState(
            x=s.x.copy(),
            P=s.P.copy(),
            S=s.S.copy(),
            gamma=s.gamma,
        )
        return BernoulliComponent(label=comp.label, r=comp.r, state=new_state)

    def _sync_state_from_best_hypothesis(self) -> None:
        """将权重最高的假设的组件拷贝到 self.state.components，供外部使用。"""
        if not self.hypotheses:
            self.state.components = []
            return
        best_hyp = max(self.hypotheses, key=lambda h: h.weight)
        self.state.components = [self._clone_component(c) for c in best_hyp.components]

    def add_target_with_state(self, x0: np.ndarray, r0: float = 0.8) -> None:
        """
        用完整真值状态初始化一个扩展目标：
        x0 = [px, py, vx, vy]；
        r0 < 1，之后由量测慢慢把 r 拉到接近 1。
        在所有现有假设中添加同一个新 label 的 Bernoulli。
        """
        label = self._new_label()

        def make_comp() -> BernoulliComponent:
            P0 = np.diag(
                [STATE_STD_POS**2, STATE_STD_POS**2,
                 (5 * STATE_STD_VEL) ** 2, (5 * STATE_STD_VEL) ** 2]
            )
            S0 = np.array([[100**2, 0], [0, 50**2]], dtype=float)
            ext_state = ExtendedTargetState(x=x0.copy(), P=P0, S=S0, gamma=10.0)
            return BernoulliComponent(label=label, r=r0, state=ext_state)

        if not self.hypotheses:
            comp = make_comp()
            self.hypotheses = [GLMBHypothesis(weight=1.0, components=[comp])]
        else:
            new_hyps: List[GLMBHypothesis] = []
            for hyp in self.hypotheses:
                comps = [self._clone_component(c) for c in hyp.components]
                comps.append(make_comp())
                new_hyps.append(GLMBHypothesis(weight=hyp.weight, components=comps))
            self.hypotheses = new_hyps

        self._sync_state_from_best_hypothesis()

    def add_target_from_position(self, pos0: np.ndarray, r0: float = 0.7) -> None:
        """
        只有位置时的初始化（速度设为 0，由量测逐渐校正）。
        动态仿真里可以用量测质心做这种初始化。
        """
        label = self._new_label()

        def make_comp() -> BernoulliComponent:
            ext_state = ExtendedTargetState.init_from_position(pos0)
            return BernoulliComponent(label=label, r=r0, state=ext_state)

        if not self.hypotheses:
            comp = make_comp()
            self.hypotheses = [GLMBHypothesis(weight=1.0, components=[comp])]
        else:
            new_hyps: List[GLMBHypothesis] = []
            for hyp in self.hypotheses:
                comps = [self._clone_component(c) for c in hyp.components]
                comps.append(make_comp())
                new_hyps.append(GLMBHypothesis(weight=hyp.weight, components=comps))
            self.hypotheses = new_hyps

        self._sync_state_from_best_hypothesis()

    # ---------- GLMB 预测 ----------

    def predict(self) -> None:
        """对所有 GLMB 假设中的 Bernoulli 做预测，并按生存概率轻微衰减 r。"""
        for hyp in self.hypotheses:
            for comp in hyp.components:
                comp.state.predict()
                comp.r *= SURVIVAL_PROB

    # ---------- GLMB 更新：对每个假设枚举多种关联方案 ----------

    def _update_single_hypothesis(
        self, hyp: GLMBHypothesis, meas_k: List[Measurement], t: float) -> List[GLMBHypothesis]:
        """
        对单个 GLMB 假设做一次 δ-GLMB 风格的更新（简化版）：

        记当前假设为 c，权重为 w^(c)，其中包含 n_comp 条 Bernoulli 轨迹。
        给定当前帧的量测集合 Z_k，我们对每个量测 z ∈ Z_k 枚举两类情况：
            1) z 由某条轨迹 ℓ 生成（被检测到），概率 ~ P_D * N(z | H x_ℓ, R)
            2) z 为杂波，概率 ~ P_FA
        并在整条量测序列上对所有量测的组合累乘似然，得到一组新的全局
        关联假设 c' = (c, θ)，对应新的 GLMBHypothesis。

        对每条轨迹，如果在某个 c' 中没有获得量测（漏检）：
            - 干扰区内：只保留生存概率（predict 阶段已经乘了 SURVIVAL_PROB）
            - 普通区域：再额外乘 (1 - P_D)，表示“该帧未被检测到”。

        这是 δ-GLMB 更新中权重项
            w^(c,θ) ∝ w^(c) ∏_ℓ η_ℓ^(c)(θ(ℓ))
        的一个数值近似实现。
        """
        # 只考虑非明显杂波（data_gen 已经打过 is_clutter 标签的）：
        non_clutter = [m for m in meas_k if not m.is_clutter]
        n_comp = len(hyp.components)

        # 没有 track 或没有量测，直接返回原假设
        if n_comp == 0 or len(non_clutter) == 0:
            return [hyp]

        # partials 里每个元素：(未归一化权重, 每条轨迹对应的量测列表)
        # assign: { track_index -> [z1, z2, ...] }
        partials: List[tuple[float, Dict[int, List[np.ndarray]]]] = [
            (hyp.weight, {i: [] for i in range(n_comp)})
        ]

        # 控制“分支数”防爆炸，比 max_hypotheses 略大一点
        max_partials = self.max_hypotheses * 4
        # 把 gate 半径的 1/3 当成 1σ，用来近似 measurement likelihood
        sigma2 = (GATING_THRESHOLD ** 2) / 9.0

        for m in non_clutter:
            z = m.z

            # 先找出在 gate 内的候选轨迹
            cand: List[tuple[int, float]] = []
            for ci, comp in enumerate(hyp.components):
                pos_pred = comp.state.x[:2]
                dist = np.linalg.norm(z - pos_pred)
                if dist <= GATING_THRESHOLD:
                    cand.append((ci, dist))

            # 如果对所有轨迹都不在 gate 内，就只产生“杂波分支”
            if not cand:
                new_partials: List[tuple[float, Dict[int, List[np.ndarray]]]] = []
                for w_old, assign_old in partials:
                    assign_new = {k: v.copy() for k, v in assign_old.items()}
                    # 纯杂波：只在权重上乘一个 P_FA
                    new_partials.append((w_old * self.P_FA, assign_new))
                partials = new_partials
                continue

            # gate 内有候选：只保留最近的 1~2 个目标，模拟 Adaptive-GLMB 里
            # k-best association 的剪枝思想
            cand.sort(key=lambda x: x[1])
            options = [cand[0]]
            if len(cand) > 1 and cand[1][1] - cand[0][1] < ASSIGN_DIST_MARGIN:
                options.append(cand[1])

            new_partials: List[tuple[float, Dict[int, List[np.ndarray]]]] = []
            for w_old, assign_old in partials:
                # (1) 杂波分支：这个量测谁都不更新
                assign_clutter = {k: v.copy() for k, v in assign_old.items()}
                new_partials.append((w_old * self.P_FA, assign_clutter))

                # (2) 检测分支：把该量测分配给候选轨迹之一
                for ci, dist in options:
                    like = float(np.exp(-0.5 * (dist ** 2) / max(sigma2, 1.0)))
                    assign_det = {k: v.copy() for k, v in assign_old.items()}
                    assign_det[ci] = assign_det[ci] + [z]
                    new_partials.append((w_old * self.P_D * like, assign_det))

            # 控制分支数量，保留权重最大的若干条
            if len(new_partials) > max_partials:
                new_partials.sort(key=lambda x: x[0], reverse=True)
                new_partials = new_partials[:max_partials]

            partials = new_partials

        # 根据 partials 生成新的 GLMB 假设，并对每条轨迹做 Kalman/扩展更新
        children: List[GLMBHypothesis] = []
        for w, assign in partials:
            # 深拷贝 track 状态，避免不同假设互相污染
            comps = [self._clone_component(c) for c in hyp.components]

            for ci, comp in enumerate(comps):
                Z_list = assign[ci]
                if len(Z_list) == 0:
                    # 该轨迹在该假设中本帧漏检
                    pos_pred = comp.state.x[:2]
                    if in_jam_region(pos_pred, t):
                        # 干扰区：仅保留 predict 阶段的生存概率，不额外乘 (1 - P_D)
                        # comp.r 已在 predict() 中乘过 SURVIVAL_PROB
                        pass
                    else:
                        # 普通区域：再乘一次 (1 - P_D)，表示这帧“应该看见但没看见”
                        comp.r *= (1.0 - self.P_D)
                else:
                    # 该轨迹在该假设下获得了 1~k 个量测（扩展目标）
                    Z = np.array(Z_list)
                    if Z.shape[0] == 1:
                        comp.state.update_with_point(Z[0], self.R_meas)
                    else:
                        comp.state.update_with_extended(Z, self.R_meas)
                    # 检测到则拉高存在概率
                    comp.r = min(0.99, comp.r + 0.1)

            children.append(GLMBHypothesis(weight=w, components=comps))

        if not children:
            return [hyp]

        return children


    def _update_hypotheses(self, meas_k: List[Measurement], t: float) -> None:
        """对当前所有 GLMB 假设做一次更新，并进行权重归一化与剪枝。"""
        if not self.hypotheses:
            return

        new_hyps: List[GLMBHypothesis] = []
        for hyp in self.hypotheses:
            new_hyps.extend(self._update_single_hypothesis(hyp, meas_k, t))

        if not new_hyps:
            # 所有假设都被“打死”，清空状态
            self.hypotheses = []
            self.state.components = []
            return

        # 1) 权重归一化
        total_w = sum(h.weight for h in new_hyps)
        if total_w > 0.0:
            for h in new_hyps:
                h.weight = h.weight / total_w

        # 2) 对每个假设内部，剪掉存在概率太低的 Bernoulli
        r_min = 1e-3
        for h in new_hyps:
            h.components = [c for c in h.components if c.r > r_min]

        # 没有任何轨迹的假设也丢掉
        new_hyps = [h for h in new_hyps if len(h.components) > 0]

        if not new_hyps:
            self.hypotheses = []
            self.state.components = []
            return

        # 3) 按权重剪枝，保留前 max_hypotheses 条假设
        new_hyps.sort(key=lambda h: h.weight, reverse=True)
        self.hypotheses = new_hyps[: self.max_hypotheses]


    # ---------- 上层接口 ----------

    def step(self, meas_k: List[Measurement], t: float) -> None:
        """
        单步递推（GLMB 版本）：
        - 对每条假设的所有轨迹做预测；
        - 针对模糊量测枚举多种关联方案，生成新的假设集合；
        - 对外暴露时取权重最大的假设作为当前 LMB 状态。
        """
        if not self.hypotheses and len(self.state.components) == 0:
            # 尚未添加任何 A/B 轨迹，直接返回
            return

        # 如果还没有初始化 GLMB 假设，但外部已经通过 state.components 加入了目标，
        # 这里将其视为一个初始假设。
        if not self.hypotheses and len(self.state.components) > 0:
            comps = [self._clone_component(c) for c in self.state.components]
            self.hypotheses = [GLMBHypothesis(weight=1.0, components=comps)]

        self.predict()
        self._update_hypotheses(meas_k, t)
        self._sync_state_from_best_hypothesis()

    def get_current_estimates(self) -> Dict[Any, np.ndarray]:
        """
        返回当前时刻各 track 的位置估计（LMB 近似）：
        - 使用权重最高的 GLMB 假设中的轨迹；
        - 只返回存在概率 r ≥ EXISTENCE_THRESHOLD 的组件；
        - Cardinality = 这些组件的个数。
        """
        est: Dict[Any, np.ndarray] = {}
        if not self.hypotheses:
            return est

        best_hyp = max(self.hypotheses, key=lambda h: h.weight)
        for comp in best_hyp.components:
            if comp.r >= EXISTENCE_THRESHOLD:
                est[comp.label] = comp.state.x[:2].copy()
        return est
