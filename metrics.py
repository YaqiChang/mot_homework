# metrics.py
import numpy as np
from typing import Dict, List, Any, Tuple
from itertools import permutations

def compute_rmse(traj_true: np.ndarray, traj_est: np.ndarray) -> float:
    """
    位置 RMSE（全程一个标量）。
    traj_true / traj_est: shape (T, 2)，其中 NaN 表示该帧无真值或无估计。
    """
    if traj_true.shape != traj_est.shape:
        raise ValueError("traj_true and traj_est must have same shape (T,2)")
    mask = ~np.isnan(traj_true[:, 0]) & ~np.isnan(traj_est[:, 0])
    if mask.sum() == 0:
        return np.nan
    diff = traj_true[mask] - traj_est[mask]
    mse = np.mean(np.sum(diff**2, axis=1))
    return float(np.sqrt(mse))

def compute_rmse_time_series(
    true_traj_dict: Dict[str, np.ndarray],
    est_traj_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    逐帧 RMSE:
        - 对每一帧 k，遍历所有目标 ID（例如 "A","B"），
          对存在真值且有估计的目标计算误差；
        - 返回:
            rmse_all: shape (T,) 每帧整体 RMSE（所有目标一起算）；
            rmse_per_id: dict[id] -> shape (T,) 每个目标自身的 RMSE；
            valid_counts: shape (T,) 每帧参与 RMSE 计算的目标个数。
    """
    ids = list(true_traj_dict.keys())
    T = next(iter(true_traj_dict.values())).shape[0]
    rmse_all = np.zeros(T, dtype=float)
    valid_counts = np.zeros(T, dtype=int)
    rmse_per_id: Dict[str, np.ndarray] = {tid: np.zeros(T, dtype=float) for tid in ids}

    for k in range(T):
        sq_list = []
        for tid in ids:
            true_traj = true_traj_dict[tid]
            est_traj = est_traj_dict.get(tid, None)
            if est_traj is None:
                continue
            if np.isnan(true_traj[k, 0]) or np.isnan(est_traj[k, 0]):
                continue
            diff = true_traj[k, :2] - est_traj[k, :2]
            sq = float(np.sum(diff ** 2))
            rmse_per_id[tid][k] = np.sqrt(sq)
            sq_list.append(sq)
        if sq_list:
            rmse_all[k] = np.sqrt(np.mean(sq_list))
            valid_counts[k] = len(sq_list)
        else:
            rmse_all[k] = 0.0
            valid_counts[k] = 0
    return rmse_all, rmse_per_id, valid_counts


def align_labels_to_truth(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],   # {"A": (T,2), "B": (T,2)}
    est_series: List[Dict[Any, np.ndarray]]  # 每步: {label: pos}
) -> Dict[Any, str]:
    """
    静态的轨迹-真值对齐：对每个 label，在整个时间维度上计算它与各真值轨迹
    的平均距离，取最小对应关系，得到 label -> "A"/"B"/...
    主要用于：
        - 离线 RMSE 统计；
        - 给可视化标注 ID 所属真值目标。
    """
    labels = set()
    for est in est_series:
        labels |= set(est.keys())
    labels = list(labels)

    label2truth: Dict[Any, str] = {}
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
            mean_d = float(np.mean(dists))
            if mean_d < best_dist:
                best_dist = mean_d
                best_id = tid
        if best_id is not None:
            label2truth[lab] = best_id

    return label2truth

def build_id_assignment_series(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],
    est_series: List[Dict[Any, np.ndarray]],
) -> List[Dict[Any, str]]:
    """
    逐帧的 label -> truth_id 近邻匹配结果，用于做 ID 时间轴图。

    返回:
        assign_series: 长度 T 的列表，
            assign_series[k] 是一个 dict: {label: truth_id 或 "None"}
            若该帧某 label 与任何真值距离都大于阈值，则 truth_id 置为 "None"。
    """
    T = len(times)
    ids = list(true_traj_dict.keys())
    assign_series: List[Dict[Any, str]] = []

    # 一个相对保守的距离阈值，用于判断“这个 label 是否真的在跟踪某个真值”
    # 这里先取 10km，可按需要调整
    dist_threshold = 10000.0

    for k in range(T):
        frame_assign: Dict[Any, str] = {}
        est_k = est_series[k]

        # 构造当帧真值列表
        truth_points = []
        truth_ids = []
        for tid in ids:
            pt = true_traj_dict[tid][k, :2]
            if not np.isnan(pt[0]):
                truth_points.append(pt)
                truth_ids.append(tid)
        truth_points_arr = (
            np.array(truth_points, dtype=float) if truth_points else np.zeros((0, 2))
        )

        for lab, pos in est_k.items():
            if truth_points_arr.shape[0] == 0:
                frame_assign[lab] = "None"
                continue
            dists = np.linalg.norm(truth_points_arr - pos[None, :], axis=1)
            j = int(np.argmin(dists))
            if dists[j] <= dist_threshold:
                frame_assign[lab] = truth_ids[j]
            else:
                frame_assign[lab] = "None"
        assign_series.append(frame_assign)

    return assign_series


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
        return float(c)
    if n == 0 and m > 0:
        return float(c)

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
    return float(ospa)


def compute_ospa_over_time(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],  # {"A": (T,2), "B": (T,2)}
    est_series: List[Dict[Any, np.ndarray]],
    p: int = 2,
    c: float = 1000.0,
    return_series: bool = False,
):
    """
    计算整个时序上的平均 OSPA 距离：
    - 每一帧构造真实目标集合 X_k、估计目标集合 Y_k；
    - 使用 _ospa_single_timestep 计算 OSPA_k；
    - 返回时间平均的 OSPA。

    若 return_series=True，则返回 (mean_ospa, ospa_vals)。
    兼容原有 main.py 只取标量的用法。
    """
    T = len(times)
    ospa_vals = np.zeros(T, dtype=float)

    trajA_true = true_traj_dict.get("A")
    trajB_true = true_traj_dict.get("B")

    for k in range(T):
        X_list = []
        if trajA_true is not None and not np.isnan(trajA_true[k, 0]):
            X_list.append(trajA_true[k, :2])
        if trajB_true is not None and not np.isnan(trajB_true[k, 0]):
            X_list.append(trajB_true[k, :2])
        X = np.array(X_list, dtype=float) if len(X_list) > 0 else np.zeros((0, 2))

        est_k = est_series[k]
        Y_list = [pos for _, pos in est_k.items()]
        Y = np.array(Y_list, dtype=float) if len(Y_list) > 0 else np.zeros((0, 2))

        ospa_vals[k] = _ospa_single_timestep(X, Y, p=p, c=c)

    mean_ospa = float(np.mean(ospa_vals))
    if return_series:
        return mean_ospa, ospa_vals
    return mean_ospa


def compute_cardinality_series(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],
    est_series: List[Dict[Any, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    逐帧基数统计：
        返回 (true_counts, est_counts, errors) 均为 shape (T,) 的数组。
    """
    T = len(times)
    trajA_true = true_traj_dict.get("A")
    trajB_true = true_traj_dict.get("B")

    true_counts = np.zeros(T, dtype=int)
    est_counts = np.zeros(T, dtype=int)
    errors = np.zeros(T, dtype=float)

    for k in range(T):
        true_c = 0
        if trajA_true is not None and not np.isnan(trajA_true[k, 0]):
            true_c += 1
        if trajB_true is not None and not np.isnan(trajB_true[k, 0]):
            true_c += 1
        est_c = len(est_series[k])
        true_counts[k] = true_c
        est_counts[k] = est_c
        errors[k] = est_c - true_c

    return true_counts, est_counts, errors

def compute_cardinality_error(
    times: np.ndarray,
    true_traj_dict: Dict[str, np.ndarray],   # {"A": (T,2), "B": (T,2)}
    est_series: List[Dict[Any, np.ndarray]],
) -> Tuple[float, float]:
    """
    基数误差统计（全程一个标量）：
    - 对每一帧 k，真实目标数 = 当帧存在的 A/B 个数；
    - 估计目标数 = est_series[k] 中 track 的个数；
    - 误差 e_k = (#est_k - #true_k)

    返回:
        mean_abs_err: 平均 |e_k|
        bias:         平均 e_k  ( >0: 经常多报; <0: 经常漏报 )
    """
    _, _, errors = compute_cardinality_series(times, true_traj_dict, est_series)
    mean_abs_err = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    return mean_abs_err, bias


def compute_detection_recall_over_time(
    true_traj_dict: Dict[str, np.ndarray],
    est_traj_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    逐帧检测召回率曲线（累积方式）：
        - 对每个目标 ID（如 "A","B"），统计到当前帧为止：
            recall_id[k] = (累计有估计的有效帧数) / (累计存在的帧数)
    返回:
        recalls: dict[id] -> shape (T,) 的 array
    """
    ids = list(true_traj_dict.keys())
    T = next(iter(true_traj_dict.values())).shape[0]
    recalls: Dict[str, np.ndarray] = {}

    for tid in ids:
        true_traj = true_traj_dict[tid]
        est_traj = est_traj_dict.get(tid, None)
        if est_traj is None:
            recalls[tid] = np.zeros(T, dtype=float)
            continue

        exist = ~np.isnan(true_traj[:, 0])
        det = exist & (~np.isnan(est_traj[:, 0]))
        cum_exist = np.cumsum(exist.astype(int))
        cum_det = np.cumsum(det.astype(int))

        recall = np.zeros(T, dtype=float)
        mask = cum_exist > 0
        recall[mask] = cum_det[mask] / cum_exist[mask]
        recalls[tid] = recall

    return recalls


def compute_confusion_matrix(
    trajA_true: np.ndarray,
    trajB_true: np.ndarray,
    estA: np.ndarray,
    estB: np.ndarray,
) -> np.ndarray:
    """
    计算一个简单的“ID 混淆矩阵”：
    - 行:  true 目标 (A, B)
    - 列:  预测为 A, 预测为 B, 未检测到 (None)

    对每一帧 k:
        若 true A 存在:
            若 estA[k] 有效 → 计为 pred=A
            elif estB[k] 有效 → 计为 pred=B (ID 反了)
            else → 计为 pred=None (漏检)
        B 同理。

    返回:
        C: shape (2, 3) 的整数矩阵:
           C[0,:] 对应 true A,  C[1,:] 对应 true B
           列顺序为 [pred A, pred B, pred None]。
    """
    idx = {"A": 0, "B": 1}
    jdx = {"A": 0, "B": 1, "None": 2}

    C = np.zeros((2, 3), dtype=int)

    T = trajA_true.shape[0]
    for k in range(T):
        # true A
        if not np.isnan(trajA_true[k, 0]):
            if not np.isnan(estA[k, 0]):
                pred = "A"
            elif not np.isnan(estB[k, 0]):
                pred = "B"
            else:
                pred = "None"
            C[idx["A"], jdx[pred]] += 1

        # true B
        if not np.isnan(trajB_true[k, 0]):
            if not np.isnan(estB[k, 0]):
                pred = "B"
            elif not np.isnan(estA[k, 0]):
                pred = "A"
            else:
                pred = "None"
            C[idx["B"], jdx[pred]] += 1

    return C
