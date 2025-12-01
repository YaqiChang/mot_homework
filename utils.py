# utils.py
import numpy as np
from typing import List, Dict
from data_gen import Measurement
from config import GATING_THRESHOLD

def group_measurements_by_radar(meas_k: List[Measurement]) -> Dict[str, List[Measurement]]:
    """按 radar_id 对量测分组"""
    radar_groups: Dict[str, List[Measurement]] = {}
    for m in meas_k:
        radar_groups.setdefault(m.radar_id, []).append(m)
    return radar_groups

def simple_gating(z: np.ndarray, predicted_pos: np.ndarray, threshold: float = GATING_THRESHOLD) -> bool:
    """简单距离 gating"""
    dist = np.linalg.norm(z - predicted_pos)
    return dist <= threshold

def cluster_extended_measurements(meas_list: List[Measurement], distance_thresh: float = 100.0):
    """
    简单的基于距离的聚类，将扩展目标多点量测分簇。（课程用简单方案就足够）
    - 输入：一个雷达的所有非杂波量测（这里暂不区分真/假）
    - 输出：若干簇，每个簇是 np.ndarray (n_i, 2)
    """
    if len(meas_list) == 0:
        return []

    points = np.array([m.z for m in meas_list])
    clusters = []

    used = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        if used[i]:
            continue
        center = points[i]
        cluster_idx = np.where(np.linalg.norm(points - center, axis=1) < distance_thresh)[0]
        used[cluster_idx] = True
        clusters.append(points[cluster_idx])

    return clusters
