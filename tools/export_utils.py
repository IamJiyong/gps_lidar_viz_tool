# tools/export_utils.py
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from lidar_gnss.io_utils import LidarScan
from lidar_gnss.pose_utils import build_interpolators


def export_synced_gps(df_gps: pd.DataFrame, scans: List[LidarScan], offset_ms: float, out_csv_path: str) -> Tuple[List[int], int]:
    """Export GPS poses synchronized to LiDAR timestamps.

    - offset_ms: LiDAR time + (-offset_ms/1000) is the GPS query time used.
    - Returns (excluded_indices, written_rows)
    """
    if df_gps is None or scans is None or len(scans) == 0:
        return ([], 0)

    interps = build_interpolators(df_gps, verbose=False)
    t_min = float(df_gps["t"].iloc[0])
    t_max = float(df_gps["t"].iloc[-1])
    dt = -float(offset_ms) * 1e-3

    cols = list(df_gps.columns)
    # enforce exact output header like original Odom_data.csv order
    ordered_cols = [
        "index","sec","nsec","pos_x","pos_y","pos_z","ori_x","ori_y","ori_z","ori_w",
        "vel_x","vel_y","vel_z","pos_cov_x","pos_cov_y","pos_cov_z",
        "pos_cov_roll","pos_cov_pitch","pos_cov_yaw","vel_cov_x","vel_cov_y","vel_cov_z"
    ]

    rows = []
    excluded: List[int] = []
    for s in scans:
        q_t = float(s.t) + dt

        # positions: interp1d가 외삽 허용
        px = float(interps.pos_interp_x([q_t])[0])
        py = float(interps.pos_interp_y([q_t])[0])
        pz = float(interps.pos_interp_z([q_t])[0])

        # orientation: SLERP는 외삽 불가 → 가장 가까운 끝으로 클램프
        q_t_clamped = float(np.clip(q_t, t_min, t_max))
        q = interps.slerp([q_t_clamped]).as_quat()[0]
        q = q / (np.linalg.norm(q) + 1e-12)

        cov_row = df_gps.iloc[-1]
        row = {
            "index": int(s.index),
            "sec": int(round(float(str(s.t).split('.')[0]))),
            "nsec": 0,
            "pos_x": px, "pos_y": py, "pos_z": pz,
            "ori_x": float(q[0]), "ori_y": float(q[1]), "ori_z": float(q[2]), "ori_w": float(q[3]),
            "vel_x": float(df_gps.get("vel_x", pd.Series([0])).iloc[0]) if "vel_x" in df_gps else 0.0,
            "vel_y": float(df_gps.get("vel_y", pd.Series([0])).iloc[0]) if "vel_y" in df_gps else 0.0,
            "vel_z": float(df_gps.get("vel_z", pd.Series([0])).iloc[0]) if "vel_z" in df_gps else 0.0,
            "pos_cov_x": float(cov_row.get("pos_cov_x", 0.0)),
            "pos_cov_y": float(cov_row.get("pos_cov_y", 0.0)),
            "pos_cov_z": float(cov_row.get("pos_cov_z", 0.0)),
            "pos_cov_roll": float(cov_row.get("pos_cov_roll", 0.0)),
            "pos_cov_pitch": float(cov_row.get("pos_cov_pitch", 0.0)),
            "pos_cov_yaw": float(cov_row.get("pos_cov_yaw", 0.0)),
            "vel_cov_x": float(cov_row.get("vel_cov_x", 0.0)),
            "vel_cov_y": float(cov_row.get("vel_cov_y", 0.0)),
            "vel_cov_z": float(cov_row.get("vel_cov_z", 0.0)),
        }
        ts = s.t
        sec = int(ts)
        nsec = int(round((ts - sec) * 1e9))
        row["sec"] = sec
        row["nsec"] = nsec
        rows.append(row)

    if not rows:
        # ensure header only file
        df_out = pd.DataFrame(columns=ordered_cols)
    else:
        df_out = pd.DataFrame(rows, columns=ordered_cols)
        df_out = df_out.sort_values("index").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    return (excluded, len(df_out)) 