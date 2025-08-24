from __future__ import annotations

import argparse
import sys
import os
from typing import Optional

import numpy as np

try:
    from lidar_gnss.io_utils import load_gnss_csv
except Exception:
    # Allow running as a script without installing the package
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from lidar_gnss.io_utils import load_gnss_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find the first timestamp where motion starts based on |velocity|. "
            "Reads Odom_data.csv (or compatible) and reports the first time speed exceeds a threshold "
            "for a sustained number of consecutive frames."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Odom_data.csv (must contain vel_{x,y}[,vel_z], pos_{x,y,z}, sec, nsec)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Speed threshold in m/s to consider as motion (default: 0.1)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=3,
        help="Require at least this many consecutive frames above threshold (default: 3)",
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D speed sqrt(vx^2+vy^2+vz^2). If not set, use planar sqrt(vx^2+vy^2)",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Optional moving-average window (odd integer) applied to speed before detection",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every k-th sample (applied before detection). Default: 1",
    )
    return parser


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    # Use 'same' to preserve length
    smoothed = np.convolve(values, kernel, mode="same")
    return smoothed


def find_first_motion_index(speed: np.ndarray, threshold: float, min_frames: int) -> Optional[int]:
    n = int(speed.shape[0])
    k = max(1, int(min_frames))
    if n == 0:
        return None
    if k == 1:
        above = np.nonzero(speed > threshold)[0]
        return int(above[0]) if above.size > 0 else None
    # Sliding window check for k consecutive above-threshold
    above_bool = speed > threshold
    run = 0
    for i in range(n):
        run = run + 1 if above_bool[i] else 0
        if run >= k:
            return i - k + 1
    return None


def main() -> int:
    args = build_arg_parser().parse_args()

    df = load_gnss_csv(args.csv, verbose=False)

    # Extract arrays
    t = df["t"].to_numpy(dtype=np.float64)
    sec = df["sec"].to_numpy(dtype=np.int64)
    nsec = df["nsec"].to_numpy(dtype=np.int64)
    vx = df["vel_x"].to_numpy(dtype=np.float64)
    vy = df["vel_y"].to_numpy(dtype=np.float64)

    if args.use_3d and "vel_z" in df.columns:
        vz = df["vel_z"].to_numpy(dtype=np.float64)
    else:
        vz = None

    # Apply stride
    stride = max(1, int(args.stride))
    t = t[::stride]
    sec = sec[::stride]
    nsec = nsec[::stride]
    vx = vx[::stride]
    vy = vy[::stride]
    if vz is not None:
        vz = vz[::stride]

    # Compute speed magnitude
    if vz is None:
        speed = np.sqrt(vx * vx + vy * vy)
    else:
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    # Optional smoothing
    speed_smoothed = moving_average(speed, window_size=int(args.smooth_window))

    idx = find_first_motion_index(
        speed=speed_smoothed, threshold=float(args.threshold), min_frames=int(args.min_frames)
    )

    if idx is None:
        print("No motion above threshold detected.")
        return 2

    # Report
    motion_t = float(t[idx])
    motion_sec = int(sec[idx])
    motion_nsec = int(nsec[idx])
    motion_speed = float(speed[idx])

    # Print a concise, parse-friendly line and a human-friendly line
    print(f"first_motion_idx={idx}, sec={motion_sec}, nsec={motion_nsec}, t={motion_t:.9f}, speed={motion_speed:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main()) 