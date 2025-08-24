from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from lidar_gnss.io_utils import LidarScan, read_lidar_bin
from lidar_gnss.pose_utils import evaluate_pose, pose_to_matrix, transform_points


@dataclass
class AccumulationResult:
    points_xyz: np.ndarray  # (M,3)
    intensities: Optional[np.ndarray]  # (M,) or None
    num_scans_used: int
    total_points_before_trim: int
    was_trimmed: bool
    per_scan_offsets: List[int]


def accumulate_lidar_points(
    scans: Sequence[LidarScan],
    t_min: float,
    t_max: float,
    time_offset: float,
    interps,
    T_base_link_lidar: np.ndarray,
    T_world_adjust: np.ndarray,
    max_points: Optional[int],
    x_range: float,
    y_range: float,
    z_range: float,
    verbose: bool,
) -> AccumulationResult:
    rng = np.random.default_rng(0)

    accumulated_xyz: Optional[np.ndarray] = None
    accumulated_intensity: Optional[np.ndarray] = None
    num_scans_used = 0
    total_points_before_trim = 0
    per_scan_offsets: List[int] = []
    cursor = 0

    for scan in tqdm(scans, desc="Accumulating LiDAR", unit="scan", disable=not verbose):
        t_adj = scan.t + time_offset
        if t_adj < t_min or t_adj > t_max:
            continue

        try:
            p_map_bl, q_map_bl = evaluate_pose(interps, t_adj)
        except Exception:
            continue

        T_map_bl = pose_to_matrix(p_map_bl, q_map_bl)
        T_map_lidar = T_world_adjust @ T_map_bl @ T_base_link_lidar

        try:
            pts, intens = read_lidar_bin(scan.path)
        except Exception:
            continue

        pts_map = transform_points(pts, T_map_lidar)

        if not (np.isinf(x_range) and np.isinf(y_range) and np.isinf(z_range)):
            mask = (
                (pts_map[:, 0] >= -x_range)
                & (pts_map[:, 0] <= x_range)
                & (pts_map[:, 1] >= -y_range)
                & (pts_map[:, 1] <= y_range)
                & (pts_map[:, 2] >= -z_range)
                & (pts_map[:, 2] <= z_range)
            )
            pts_map = pts_map[mask]
            if intens is not None:
                intens = intens[mask]

        if pts_map.shape[0] == 0:
            continue

        num_scans_used += 1
        total_points_before_trim += pts_map.shape[0]

        per_scan_offsets.append(cursor)
        if accumulated_xyz is None:
            accumulated_xyz = pts_map
            accumulated_intensity = intens
        else:
            accumulated_xyz = np.concatenate([accumulated_xyz, pts_map], axis=0)
            if accumulated_intensity is not None and intens is not None:
                accumulated_intensity = np.concatenate([accumulated_intensity, intens], axis=0)
            else:
                accumulated_intensity = None
        cursor += pts_map.shape[0]

    if accumulated_xyz is None:
        return AccumulationResult(
            points_xyz=np.zeros((0, 3), dtype=np.float64),
            intensities=None,
            num_scans_used=0,
            total_points_before_trim=0,
            was_trimmed=False,
            per_scan_offsets=[],
        )

    was_trimmed = False
    if max_points is not None and accumulated_xyz.shape[0] > max_points:
        was_trimmed = True
        idx = rng.choice(accumulated_xyz.shape[0], size=max_points, replace=False)
        accumulated_xyz = accumulated_xyz[idx]
        if accumulated_intensity is not None:
            accumulated_intensity = accumulated_intensity[idx]

    return AccumulationResult(
        points_xyz=accumulated_xyz,
        intensities=accumulated_intensity,
        num_scans_used=num_scans_used,
        total_points_before_trim=total_points_before_trim,
        was_trimmed=was_trimmed,
        per_scan_offsets=per_scan_offsets,
    ) 