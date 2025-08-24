from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore

from lidar_gnss.common import _log


def load_extrinsics(path: Optional[str], verbose: bool) -> np.ndarray:
    if not path:
        _log("No extrinsics provided; using identity for T_base_link_lidar.", verbose)
        return np.eye(4, dtype=np.float64)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Extrinsics file not found: {path}")

    try:
        with open(path, "r") as f:
            text = f.read()
        data: Dict[str, object]
        if path.lower().endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("pyyaml is required to read YAML extrinsics.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        if isinstance(data, dict) and "T_base_link_lidar" in data:
            T_list = data["T_base_link_lidar"]  # type: ignore[index]
            T = np.array(T_list, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError("T_base_link_lidar must be a 4x4 matrix.")
            return T
        else:
            _log("Extrinsics file missing 'T_base_link_lidar'; using identity.", verbose)
            return np.eye(4, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse extrinsics from {path}: {exc}")


def load_gnss_csv(csv_path: str, verbose: bool) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"GNSS CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "index",
        "sec",
        "nsec",
        "pos_x",
        "pos_y",
        "pos_z",
        "ori_x",
        "ori_y",
        "ori_z",
        "ori_w",
        "vel_x",
        "vel_y",
        "vel_z",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["t"] = df["sec"].astype(np.float64) + df["nsec"].astype(np.float64) * 1e-9
    df = df.sort_values("t", ascending=True).reset_index(drop=True)

    t_vals = df["t"].to_numpy()
    if np.any(~np.isfinite(t_vals)):
        raise ValueError("Non-finite timestamps found in CSV.")
    if np.any(np.diff(t_vals) <= 0):
        raise ValueError("Timestamps are not strictly increasing (duplicates/backwards).")

    q = df[["ori_x", "ori_y", "ori_z", "ori_w"]].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(q)):
        raise ValueError("NaNs/Infs in quaternion columns.")
    # Normalize per-row
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q = q / norms
    df.loc[:, ["ori_x", "ori_y", "ori_z", "ori_w"]] = q

    p = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise ValueError("NaNs/Infs in position columns.")

    _log(
        f"Loaded GNSS CSV with {len(df)} poses from t=[{t_vals[0]:.3f}, {t_vals[-1]:.3f}] (s)",
        verbose,
    )
    return df


LIDAR_FILENAME_RE = re.compile(r"^lidar_(?P<index>\d{6,})_(?P<sec>\d{1,})_(?P<nsec>\d{1,9})xyzi\.bin$")


@dataclass
class LidarScan:
    path: str
    index: int
    t: float


def parse_lidar_directory(lidar_dir: str) -> List[LidarScan]:
    if not os.path.isdir(lidar_dir):
        raise NotADirectoryError(f"LiDAR directory not found: {lidar_dir}")

    scans: List[LidarScan] = []
    for name in os.listdir(lidar_dir):
        m = LIDAR_FILENAME_RE.match(name)
        if not m:
            continue
        sec = int(m.group("sec"))
        nsec = int(m.group("nsec"))
        idx = int(m.group("index"))
        t = float(sec) + float(nsec) * 1e-9
        scans.append(LidarScan(path=os.path.join(lidar_dir, name), index=idx, t=t))

    if not scans:
        raise RuntimeError(
            "No LiDAR scans found. Ensure filenames follow 'lidar_{index}_{sec}_{nanosec}.bin'."
        )

    scans.sort(key=lambda s: (s.t, s.index))

    times = np.array([s.t for s in scans], dtype=np.float64)
    if np.any(np.diff(times) < 0):
        raise RuntimeError("LiDAR scan times are not non-decreasing.")

    return scans


def read_lidar_bin(path: str):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise RuntimeError(f"Invalid LiDAR bin file (size not multiple of 4): {path}")
    arr = arr.reshape(-1, 4)
    points_xyz = arr[:, :3].astype(np.float64)
    intensities = arr[:, 3].astype(np.float32)
    return points_xyz, intensities 