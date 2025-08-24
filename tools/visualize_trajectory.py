#!/usr/bin/env python3
"""
visualize_trajectory.py (refactored; LiDAR required; lidar-prep & callbacks split)

Visualize GNSS trajectory (polyline + sampled markers/arrows) and a LiDAR point cloud
transformed into the same world frame. Interactive stepping/time-offset updates via Open3D.

Controls:
  ←/→ or [/]= scan ±step; ,/. = time offset ±step; Q/Esc to exit.

GNSS CSV header:
  index,sec,nsec,pos_x,pos_y,pos_z,ori_x,ori_y,ori_z,ori_w,vel_x,vel_y,vel_z
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List, Any

import numpy as np

from lidar_gnss.io_utils import (  # type: ignore
    load_gnss_csv,
    parse_lidar_directory,
    read_lidar_bin,
    load_extrinsics,
)
from lidar_gnss.pose_utils import (  # type: ignore
    ResampledPoses,
    resample_poses,
    build_interpolators,
    evaluate_pose,
    pose_to_matrix,
    transform_points,
    update_heading_from_path,
)
from lidar_gnss.viz_open3d import visualize_scene_open3d  # unified visualizer


# ---------------------------- CLI ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize GNSS trajectory (polyline + markers/arrows) with a LiDAR frame."
    )
    # LiDAR (REQUIRED)
    p.add_argument("--lidar_dir", type=str, required=True, help="Directory with LiDAR .bin scans (required).")
    p.add_argument("--lidar_index", type=int, required=True, help="LiDAR frame index to visualize (required).")

    # GNSS
    p.add_argument("--gps_csv", type=str, required=True, help="Path to GNSS (odometry) CSV")
    p.add_argument("--target_rate", type=float, default=10.0, help="GNSS resampling rate in Hz (default: 10)")
    p.add_argument("--stride", type=int, default=5, help="Every k-th resampled pose for markers/arrows (k>=1).")

    p.add_argument(
        "--origin_from_first",
        action="store_true",
        default=True,
        help="Shift world so the first GNSS position is at (0,0,0). Enabled by default. Disable with --no-origin_from_first.",
    )
    p.add_argument("--no-origin_from_first", dest="origin_from_first", action="store_false", help="Do not shift world origin.")

    # LiDAR transform
    p.add_argument("--time_offset", type=float, default=0.0, help="Seconds added to LiDAR timestamp before GNSS interp.")
    p.add_argument(
        "--extrinsics_yaml",
        type=str,
        default=None,
        help="YAML/JSON with 4x4 T_base_link_lidar under key 'T_base_link_lidar'. If omitted, identity is used.",
    )

    # Cropping
    p.add_argument("--x_range", type=float, default=30.0, help="Half-extent meters for |x|<=x_range.")
    p.add_argument("--y_range", type=float, default=30.0, help="Half-extent meters for |y|<=y_range.")

    # Interaction
    p.add_argument("--step_size", type=int, default=5, help="Key step for scan/time offset (default: 5).")

    # Orientation source
    p.add_argument("--heading_from_pose", action="store_true", help="Use XY heading from positions instead of CSV orientations.")

    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p


# ---------------------------- Helpers ----------------------------

def _shift_origin(
    p_raw: np.ndarray,
    p_resampled: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally shift world so the FIRST raw GNSS position is at the origin."""
    T_world_adjust = np.eye(4, dtype=np.float64)
    if enabled and p_raw.shape[0] > 0:
        p0 = p_raw[0].copy()
        p_raw = p_raw - p0
        p_resampled = p_resampled - p0
        T_world_adjust[:3, 3] = -p0
    return p_raw, p_resampled, T_world_adjust


def _crop_xy(points: np.ndarray, x_range: float, y_range: float) -> np.ndarray:
    """Crop points by XY half-extents."""
    if points.size == 0 or (np.isinf(x_range) and np.isinf(y_range)):
        return points
    mask = (
        (points[:, 0] >= -x_range) & (points[:, 0] <= x_range) &
        (points[:, 1] >= -y_range) & (points[:, 1] <= y_range)
    )
    return points[mask]


@dataclass
class LidarContext:
    scans: List[Any]
    interps: Any
    T_world_adjust: np.ndarray
    T_base_link_lidar: np.ndarray
    t_min: float
    t_max: float
    x_range: float
    y_range: float
    stride: int
    resampled: ResampledPoses


@dataclass
class LidarPrepResult:
    points_world_init: np.ndarray
    highlight_marker_index: Optional[int]
    debug_msg: Optional[str]
    context: LidarContext
    initial_idx_pos: int


def _prepare_lidar_initial(
    *,
    lidar_dir: str,
    lidar_index: int,
    df,
    resampled: ResampledPoses,
    T_world_adjust: np.ndarray,
    time_offset: float,
    x_range: float,
    y_range: float,
    stride: int,
    extrinsics_yaml: Optional[str],
    verbose: bool,
) -> LidarPrepResult:
    """
    Compute initial LiDAR world-frame point cloud, highlight index, debug message, and
    build a context object for later interactive callbacks.
    """
    scans = parse_lidar_directory(lidar_dir)
    sel = [s for s in scans if s.index == int(lidar_index)]
    if not sel:
        raise ValueError(f"LiDAR index {lidar_index} not found in directory: {lidar_dir}")
    scan = sel[0]
    initial_idx_pos = scans.index(scan)

    interps = build_interpolators(df, verbose=False)
    t_min = float(df["t"].iloc[0])
    t_max = float(df["t"].iloc[-1])

    t_adj = float(scan.t) + float(time_offset)
    if t_adj < t_min or t_adj > t_max:
        raise ValueError(f"Adjusted LiDAR time {t_adj:.6f}s out of GNSS range [{t_min:.6f}, {t_max:.6f}]s.")

    p_map_bl, q_map_bl = evaluate_pose(interps, t_adj)
    T_map_bl = pose_to_matrix(p_map_bl, q_map_bl)
    T_base_link_lidar = load_extrinsics(extrinsics_yaml, verbose=verbose)
    T_map_lidar = T_world_adjust @ T_map_bl @ T_base_link_lidar

    pts, _ = read_lidar_bin(scan.path)
    pts_map = transform_points(pts, T_map_lidar)
    pts_map = _crop_xy(pts_map, x_range, y_range)

    # Time-nearest marker index for highlighting
    marker_times = resampled.times[::stride]
    highlight = int(np.argmin(np.abs(marker_times - t_adj))) if marker_times.size > 0 else None

    debug_msg = None
    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        debug_msg = (
            f"LiDAR transform debug (t_adj={t_adj:.6f} s):\n"
            f"- p_map_bl: {np.array2string(p_map_bl, precision=3, suppress_small=True)}\n"
            f"- q_map_bl (xyzw): {np.array2string(q_map_bl, precision=3, suppress_small=True)}\n"
            f"- T_world_adjust:\n{np.array2string(T_world_adjust, precision=3, suppress_small=True)}\n"
            f"- T_map_bl:\n{np.array2string(T_map_bl, precision=3, suppress_small=True)}\n"
            f"- T_base_link_lidar:\n{np.array2string(T_base_link_lidar, precision=3, suppress_small=True)}\n"
            f"- T_map_lidar = T_world_adjust @ T_map_bl @ T_base_link_lidar:\n"
            f"{np.array2string(T_map_lidar, precision=3, suppress_small=True)}\n"
        )

    context = LidarContext(
        scans=scans,
        interps=interps,
        T_world_adjust=T_world_adjust,
        T_base_link_lidar=T_base_link_lidar,
        t_min=t_min,
        t_max=t_max,
        x_range=float(x_range),
        y_range=float(y_range),
        stride=int(stride),
        resampled=resampled,
    )

    return LidarPrepResult(
        points_world_init=pts_map,
        highlight_marker_index=highlight,
        debug_msg=debug_msg,
        context=context,
        initial_idx_pos=initial_idx_pos,
    )


def _make_lidar_callbacks(
    *,
    context: LidarContext,
    initial_idx_pos: int,
    initial_time_offset: float,
) -> tuple[
    Callable[[int], Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]],
    Callable[[float], Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]],
]:
    """
    Create scan/time-offset updater callbacks using the prepared context and initial state.
    Returns:
      scan_updater(delta), time_offset_updater(dt_seconds)
    """
    current_idx_pos = int(initial_idx_pos)
    current_time_offset = float(initial_time_offset)

    def _recompute_for_current() -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        sc = context.scans[current_idx_pos]
        t_adj_loc = float(sc.t) + float(current_time_offset)
        if t_adj_loc < context.t_min or t_adj_loc > context.t_max:
            return np.zeros((0, 3), dtype=np.float64), None, None
        p_map_bl_loc, q_map_bl_loc = evaluate_pose(context.interps, t_adj_loc)
        T_map_bl_loc = pose_to_matrix(p_map_bl_loc, q_map_bl_loc)
        T_map_lidar_loc = context.T_world_adjust @ T_map_bl_loc @ context.T_base_link_lidar
        pts_loc, _intens_loc = read_lidar_bin(sc.path)
        pts_map_loc = transform_points(pts_loc, T_map_lidar_loc)
        pts_map_loc = _crop_xy(pts_map_loc, context.x_range, context.y_range)

        new_hi = None
        try:
            marker_times_loc = context.resampled.times[::context.stride]
            if marker_times_loc.size > 0:
                new_hi = int(np.argmin(np.abs(marker_times_loc - t_adj_loc)))
        except Exception:
            new_hi = None
        return pts_map_loc, None, new_hi

    def scan_updater(delta: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        nonlocal current_idx_pos
        new_pos = int(np.clip(current_idx_pos + int(delta), 0, len(context.scans) - 1))
        current_idx_pos = new_pos
        return _recompute_for_current()

    def time_offset_updater(delta_time: float) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        nonlocal current_time_offset
        current_time_offset = float(current_time_offset) + float(delta_time)
        return _recompute_for_current()

    return scan_updater, time_offset_updater


# ---------------------------- Main ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("stride must be >= 1")
    verbose = bool(args.verbose)

    # 1) Load GNSS (optionally adjust heading)
    df = load_gnss_csv(args.gps_csv, verbose=verbose)
    if args.heading_from_pose:
        df = update_heading_from_path(df, verbose=verbose)

    # Raw polyline (ALL GNSS points)
    p_raw = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)

    # 2) Resample for evenly spaced markers/arrows
    resampled: ResampledPoses = resample_poses(df, target_rate_hz=float(args.target_rate), verbose=verbose)
    p_resampled = resampled.positions
    q_resampled = resampled.quaternions

    # 3) Optional origin shift so the first raw GNSS position is at (0,0,0)
    p_raw, p_resampled, T_world_adjust = _shift_origin(p_raw, p_resampled, enabled=bool(args.origin_from_first))

    # 4) Build marker arrays with stride
    stride = int(args.stride)
    marker_points = p_resampled[::stride]
    marker_quats = q_resampled[::stride]

    # 5) LiDAR prep (initial) + separate callback construction
    prep = _prepare_lidar_initial(
        lidar_dir=args.lidar_dir,
        lidar_index=int(args.lidar_index),
        df=df,
        resampled=resampled,
        T_world_adjust=T_world_adjust,
        time_offset=float(args.time_offset),
        x_range=float(args.x_range),
        y_range=float(args.y_range),
        stride=stride,
        extrinsics_yaml=args.extrinsics_yaml,
        verbose=verbose,
    )
    scan_updater_cb, time_offset_updater_cb = _make_lidar_callbacks(
        context=prep.context,
        initial_idx_pos=prep.initial_idx_pos,
        initial_time_offset=float(args.time_offset),
    )
    if verbose and prep.debug_msg:
        print(prep.debug_msg, flush=True)

    # 6) Summary
    t_min = float(df["t"].iloc[0])
    t_max = float(df["t"].iloc[-1])
    summary = (
        "Trajectory visualization summary:\n"
        f"- Raw GNSS points (polyline): {p_raw.shape[0]}\n"
        f"- Resampled poses: {resampled.positions.shape[0]} (rate={float(args.target_rate):.3f} Hz)\n"
        f"- Markers used: {marker_points.shape[0]} (stride={stride})\n"
        f"- GNSS t-range: [{t_min:.3f}, {t_max:.3f}] s\n"
        f"- Origin shift: {'first position to (0,0,0)' if args.origin_from_first else 'disabled'}\n"
        f"- LiDAR: dir={args.lidar_dir}, index={args.lidar_index}, color_mode=z, "
        f"crop: x=[{-float(args.x_range):.1f},{float(args.x_range):.1f}] y=[{-float(args.y_range):.1f},{float(args.y_range):.1f}], "
        f"highlight_marker_index={prep.highlight_marker_index}\n"
    )

    # 7) Visualize with unified function
    visualize_scene_open3d(
        # Cloud (LiDAR always present)
        points_xyz=prep.points_world_init,
        color_mode="z",  # per-scan coloring is not applicable for a single frame
        # Trajectory
        polyline_points_full=p_raw,
        marker_points=marker_points,
        marker_quats_xyzw=marker_quats,
        highlight_marker_index=prep.highlight_marker_index,
        # Interaction
        scan_updater=scan_updater_cb,
        time_offset_updater=time_offset_updater_cb,
        key_step=int(args.step_size),
        # Window / misc
        window_title="Trajectory + LiDAR Viewer",
        summary=summary,
        print_summary=True,
    )
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
