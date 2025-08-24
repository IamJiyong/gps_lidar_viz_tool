#!/usr/bin/env python3
"""
merge_lidar_gnss.py (refactored to unified Open3D viewer)

Time-synchronize LiDAR point clouds with GNSS (odometry)+IMU pose, merge them by chunks,
and visualize with Open3D via a single consolidated function (visualize_scene_open3d).

Changes:
- Removed intensity coloring; use either per-scan colors (if valid) or z-based colors.
- Per-scan coloring auto-falls back to z coloring when offsets are invalid (e.g., after trim).
- Full GNSS polyline (origin-shifted) is drawn; markers/arrows are sampled at chunk timestamps.
- Interactive mode uses scan/time-offset callbacks wired into visualize_scene_open3d.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lidar_gnss.common import _log
from lidar_gnss.io_utils import load_extrinsics, load_gnss_csv, parse_lidar_directory
from lidar_gnss.pose_utils import build_interpolators, resample_poses
from lidar_gnss.accumulate import accumulate_lidar_points
from lidar_gnss.viz_open3d import visualize_scene_open3d


# ---------------------------- CLI ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time-synchronize LiDAR point clouds using GNSS+IMU odometry, merge, and visualize with Open3D."
    )
    parser.add_argument("--gps_csv", type=str, required=True, help="Path to GNSS (odometry) CSV")
    parser.add_argument("--lidar_dir", type=str, required=True, help="Directory with LiDAR .bin scans")

    parser.add_argument("--target_rate", type=float, default=10.0, help="GNSS interpolation grid rate (Hz)")
    parser.add_argument(
        "--time_offset",
        type=float,
        default=0.0,
        help="Time offset (nanoseconds) added to LiDAR timestamps before interpolation.",
    )
    parser.add_argument(
        "--extrinsics_yaml",
        type=str,
        default="extrinsics.yaml",
        help="YAML/JSON with 4x4 T_base_link_lidar under key 'T_base_link_lidar'. If omitted, identity is used.",
    )

    parser.add_argument("--max_points", type=int, default=None, help="Max accumulated points per chunk (trim if exceeded).")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max LiDAR scans to merge per chunk. If omitted, merge all effective scans into one chunk.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start LiDAR scan index (after sort, before subsampling).")
    parser.add_argument("--index_interval", type=int, default=1, help="Process every k-th LiDAR scan (k>=1).")

    parser.add_argument("--x_range", type=float, default=30.0, help="Keep points with |x| <= x_range (meters).")
    parser.add_argument("--y_range", type=float, default=30.0, help="Keep points with |y| <= y_range (meters).")
    parser.add_argument("--z_range", type=float, default=30.0, help="Keep points with |z| <= z_range (meters).")

    parser.add_argument(
        "--origin_from_first",
        action="store_true",
        default=True,
        help="Shift world so the first GNSS position is at (0,0,0). Enabled by default. Disable with --no-origin_from_first.",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Interactive controls (mapped to unified viewer)
    parser.add_argument("--index_step", type=int, default=1, help="Start index step for Left/Right (default: 1)")
    parser.add_argument("--offset_step_ns", type=float, default=1e7, help="Time offset step (ns) for ,/. keys (default: 1e7=10ms)")
    parser.add_argument(
        "--marker_stride",
        type=int,
        default=5,
        help="Stride for resampled markers/arrows in interactive mode (k>=1).",
    )

    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["auto", "z", "per_scan"],
        default="auto",
        help="Point color mode: 'auto' (default), 'z' (height-based), or 'per_scan' (one color per scan if available).",
    )

    return parser

# ---------------------------- Main ----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verbose = bool(args.verbose)

    # Load inputs
    df = load_gnss_csv(args.gps_csv, verbose=verbose)
    interps = build_interpolators(df, verbose=verbose)

    # Cache resample (some downstream tools might rely on it)
    _ = resample_poses(df, target_rate_hz=float(args.target_rate), verbose=verbose)

    # Extrinsics
    T_base_link_lidar = load_extrinsics(args.extrinsics_yaml, verbose=verbose)

    # LiDAR scans
    scans = parse_lidar_directory(args.lidar_dir)

    # GNSS time span
    t_min = float(df["t"].iloc[0])
    t_max = float(df["t"].iloc[-1])

    # World origin adjustment
    if bool(args.origin_from_first):
        p0 = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)[0]
        T_world_adjust = np.eye(4, dtype=np.float64)
        T_world_adjust[:3, 3] = -p0
    else:
        T_world_adjust = np.eye(4, dtype=np.float64)

    # Precompute full polyline for visualization (origin-shifted to match chunks)
    polyline_full = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    if bool(args.origin_from_first):
        polyline_full = polyline_full - polyline_full[0]

    if verbose:
        _log(f"Applying time_offset={float(args.time_offset) / 1e9:.6f} seconds to LiDAR times.", verbose)

    _run_interactive(
        df=df,
        interps=interps,
        scans=scans,
        t_min=t_min,
        t_max=t_max,
        T_base_link_lidar=T_base_link_lidar,
        T_world_adjust=T_world_adjust,
        polyline_full=polyline_full,
        args=args,
    )



# ---------------------------- Interactive path ----------------------------
def _run_interactive(
    *,
    df: pd.DataFrame,
    interps,
    scans,
    t_min: float,
    t_max: float,
    T_base_link_lidar: np.ndarray,
    T_world_adjust: np.ndarray,
    polyline_full: np.ndarray,
    args,
) -> int:
    total_scans = len(scans)
    start_index = int(args.start_index)
    k = int(args.index_interval)
    max_frames = int(args.max_frames) if args.max_frames is not None else (total_scans - start_index)

    # --- 새로 추가: 인터랙티브용 마커/화살표 준비 (전 구간 고정 표식) ---
    resampled = resample_poses(df, target_rate_hz=float(args.target_rate), verbose=False)
    stride = max(1, int(args.marker_stride))
    marker_points_full = resampled.positions[::stride].copy()
    marker_quats_full = resampled.quaternions[::stride].copy()
    # origin shift와 동일 프레임으로
    marker_points_full += T_world_adjust[:3, 3]  # T_world_adjust는 [-p0] 번역

    marker_times_sub = resampled.times[::stride]

    def build_chunk_indices(si: int) -> List[int]:
        si = max(0, min(total_scans - 1, si))
        end = total_scans
        if max_frames > 0:
            end = min(total_scans, si + max_frames * k)
        return list(range(si, end, k))

    def accumulate_for(si: int, time_offset_ns: float):
        sel_indices = build_chunk_indices(si)
        chunk_scans = [scans[i] for i in sel_indices]
        acc = accumulate_lidar_points(
            scans=chunk_scans,
            t_min=t_min,
            t_max=t_max,
            time_offset=float(time_offset_ns) / 1e9,   # ns → s
            interps=interps,
            T_base_link_lidar=T_base_link_lidar,
            T_world_adjust=T_world_adjust,
            max_points=int(args.max_points) if args.max_points is not None else None,
            x_range=float(args.x_range),
            y_range=float(args.y_range),
            z_range=float(args.z_range),
            verbose=False,
        )
        return acc, chunk_scans

    # 대표 시간으로 하이라이트 계산
    def compute_highlight(si: int, time_offset_ns: float) -> Optional[int]:
        _, chunk_scans = accumulate_for(si, time_offset_ns)
        for sc in chunk_scans:
            t_adj = float(sc.t) + float(time_offset_ns) / 1e9
            if t_min <= t_adj <= t_max and marker_times_sub.size > 0:
                return int(np.argmin(np.abs(marker_times_sub - t_adj)))
        return None

    # Initial state
    current_index = max(0, min(total_scans - 1, start_index))
    current_offset_ns = float(args.time_offset)

    acc0, _ = accumulate_for(current_index, current_offset_ns)
    if acc0.points_xyz.size == 0:
        print("No points to visualize in initial selection.")
        return 0

    initial_hi = compute_highlight(current_index, current_offset_ns)

    # --- Callbacks for unified viewer ---
    idx_step = max(1, int(args.index_step))
    offset_step_seconds = float(args.offset_step_ns) / 1e9  # ,/. 한 번당 초 단위

    def scan_updater(delta: int):
        nonlocal current_index
        current_index = int(np.clip(current_index + int(delta) * idx_step, 0, total_scans - 1))
        acc, _ = accumulate_for(current_index, current_offset_ns)
        hi = compute_highlight(current_index, current_offset_ns)
        print(f"[key] index={current_index} | offset_ns={current_offset_ns:.0f}")
        return acc.points_xyz, None, hi

    def time_offset_updater(dt: float):
        nonlocal current_offset_ns
        current_offset_ns = float(current_offset_ns) + float(dt) * offset_step_seconds * 1e9  # 내부 ns 유지
        acc, _ = accumulate_for(current_index, current_offset_ns)
        hi = compute_highlight(current_index, current_offset_ns)
        print(f"[key] index={current_index} | offset_ns={current_offset_ns:.0f}")
        return acc.points_xyz, None, hi

    # Summary for interactive session
    summary = (
        "Interactive visualization (merged chunks):\n"
        f"- Start index: {current_index} (step={idx_step})\n"
        f"- Time offset step: {offset_step_seconds:.6f} s per key press (, or .)\n"
        f"- Chunk size (max_frames): {max_frames}  | interval k: {k}\n"
        f"- GNSS t-range: [{t_min:.3f}, {t_max:.3f}] s\n"
        f"- Origin shift: {'first GNSS position to (0,0,0)' if args.origin_from_first else 'disabled'}\n"
        f"- Markers: every {stride}-th resampled pose (fixed across updates)\n"
        f"- Color mode: {args.color_mode}"
    )

    # Launch unified viewer (interactive) — 이제 마커/화살표 + 하이라이트 포함
    visualize_scene_open3d(
        points_xyz=acc0.points_xyz,
        color_mode="z",                         # 업데이트 프레임은 z 컬러
        polyline_points_full=polyline_full,     # 전체 폴리라인
        marker_points=marker_points_full,       # 고정 마커
        marker_quats_xyzw=marker_quats_full,    # 화살표 방향
        highlight_marker_index=initial_hi,      # 초기 하이라이트
        scan_updater=scan_updater,
        time_offset_updater=time_offset_updater,
        key_step=1,                             # viewer는 ±1 전달 → 내부에서 idx_step/offset 변환
        window_title="LiDAR-GNSS Merge (Interactive)",
        summary=summary,
        print_summary=True,
    )

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
