# metrics.py
import numpy as np
from typing import Dict, List, Any
from itertools import permutations
def compute_rmse(traj_true: np.ndarray, traj_est: np.ndarray) -> float:
    """
    简单位置 RMSE，traj_true/est shape (T, 2)，其中 nan 表示无估计。
    """
    mask = ~np.isnan(traj_true[:, 0]) & ~np.isnan(traj_est[:, 0])
    if mask.sum() == 0:
        return np.nan
    diff = traj_true[mask] - traj_est[mask]
    mse = np.mean(np.sum(diff**2, axis=1))
    return np.sqrt(mse)

def align_labels_to_truth(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],   # {"A": (T,2), "B": (T,2)}
    est_series: List[Dict[Any, np.ndarray]]  # 每步: {label: pos}
):
    """
    非严格的轨迹-真值对齐：按近邻在整个时间上match label -> A/B，
    用于大概统计 RMSE 和 ID-switch。
    这里只给一个简单示意（完整 ID-switch 统计你可以再精细化）。
    """
    labels = set()
    for est in est_series:
        labels |= set(est.keys())
    labels = list(labels)

    # 对每个 label 和每个真值ID，计算平均距离
    label2truth = {}
    for lab in labels:
        best_id = None
        best_dist = np.inf
        for tid, traj_true in true_traj_dict.items():
            dists = []
            for k, est in enumerate(est_series):
                if lab in est and not np.isnan(traj_true[k, 0]):
                    d = np.linalg.norm(est[lab] - traj_true[k])
                    dists.append(d)
            if len(dists) == 0:
                continue
            mean_d = np.mean(dists)
            if mean_d < best_dist:
                best_dist = mean_d
                best_id = tid
        if best_id is not None:
            label2truth[lab] = best_id

    return label2truth


def _ospa_single_timestep(
    X: np.ndarray,
    Y: np.ndarray,
    p: int = 2,
    c: float = 1000.0
) -> float:
    """
    单时刻 OSPA 距离（简化实现）:
    X, Y: (m,2)/(n,2) 的二维位置集合
    p: 距离幂次（一般取 1 或 2）
    c: 截断距离（超过 c 的误差按 c 计）
    """
    m = X.shape[0]
    n = Y.shape[0]

    if m == 0 and n == 0:
        return 0.0

    # 只有一边有目标：纯基数误差
    if m == 0 and n > 0:
        return c
    if n == 0 and m > 0:
        return c

    # 计算截断距离矩阵 d_ij
    D = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(X[i] - Y[j])
            D[i, j] = min(c, d)

    # 按 OSPA 定义，使用 max(m, n) 作为归一化因子
    N = max(m, n)

    # 枚举较小一侧的排列，找到最小匹配代价
    if m <= n:
        # 为每个真实目标寻找唯一对应的估计目标
        min_cost = np.inf
        for perm in permutations(range(n), m):
            cost = 0.0
            for i in range(m):
                cost += D[i, perm[i]] ** p
            # 未匹配的估计目标作为虚警，按 c^p 计
            cost += (n - m) * (c ** p)
            if cost < min_cost:
                min_cost = cost
    else:
        # 估计少于真实目标时，对称处理
        min_cost = np.inf
        for perm in permutations(range(m), n):
            cost = 0.0
            for j in range(n):
                cost += D[perm[j], j] ** p
            # 未匹配的真实目标作为漏检，按 c^p 计
            cost += (m - n) * (c ** p)
            if cost < min_cost:
                min_cost = cost

    ospa_p = min_cost / N
    ospa = ospa_p ** (1.0 / p)
    return ospa


def compute_ospa_over_time(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],  # {"A": (T,2), "B": (T,2)}
    est_series: List[Dict[Any, np.ndarray]],
    p: int = 2,
    c: float = 1000.0
) -> float:
    """
    计算整个时序上的平均 OSPA 距离：
    - 每一帧构造真实目标集合 X_k、估计目标集合 Y_k；
    - 使用 _ospa_single_timestep 计算 OSPA_k；
    - 返回时间平均的 OSPA。
    这里不使用 label 对齐，而是直接把当前帧所有估计位置当作集合 Y_k。
    """
    T = len(times)
    ospa_vals = []

    trajA_true = true_traj_dict["A"]   # (T,2)
    trajB_true = true_traj_dict["B"]   # (T,2)

    for k in range(T):
        # 构造真实集合 X_k：只加入存在的目标
        X_list = []
        if not np.isnan(trajA_true[k, 0]):
            X_list.append(trajA_true[k])
        if not np.isnan(trajB_true[k, 0]):
            X_list.append(trajB_true[k])
        X = np.array(X_list, dtype=float) if len(X_list) > 0 else np.zeros((0, 2))

        # 构造估计集合 Y_k：当前帧所有估计的目标位置
        est_k = est_series[k]
        Y_list = [pos for _, pos in est_k.items()]
        Y = np.array(Y_list, dtype=float) if len(Y_list) > 0 else np.zeros((0, 2))

        d_k = _ospa_single_timestep(X, Y, p=p, c=c)
        ospa_vals.append(d_k)

    ospa_vals = np.array(ospa_vals, dtype=float)
    mean_ospa = float(np.mean(ospa_vals))
    return mean_ospa