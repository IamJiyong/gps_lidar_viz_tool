# lidar_gnss/viz_open3d.py
from __future__ import annotations
import colorsys
from typing import Callable, List, Optional, Tuple

import numpy as np
import open3d as o3d


def _ensure_open3d_installed() -> None:
    if o3d is None:
        raise ImportError("open3d is required")


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in q]
    n = x*x + y*y + z*z + w*w
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    R = np.array([
        [1.0 - (yy + zz),       xy - wz,            xz + wy],
        [      xy + wz,   1.0 - (xx + zz),          yz - wx],
        [      xz - wy,         yz + wx,      1.0 - (xx + yy)],
    ], dtype=np.float64)
    return R


def visualize_scene_open3d(
    *,
    # --- Cloud ---
    points_xyz: Optional[np.ndarray] = None,            # (N,3) float64
    color_mode: str = "z",                              # "z" | "per_scan"
    per_scan_offsets: Optional[List[int]] = None,
    per_scan_colors: Optional[List[Tuple[float, float, float]]] = None,

    # --- Trajectory / Markers ---
    polyline_points_full: Optional[np.ndarray] = None,  # (M,3)
    marker_points: Optional[np.ndarray] = None,         # (K,3)
    marker_quats_xyzw: Optional[np.ndarray] = None,     # (K,4) quats in xyzw
    highlight_marker_index: Optional[int] = None,

    # --- Interaction (callbacks kept as-is) ---
    scan_updater: Optional[Callable[[int], Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]]] = None,
    time_offset_updater: Optional[Callable[[float], Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]]] = None,
    key_step: int = 5,

    # --- Style & Behavior ---
    show_axes: bool = True,
    axes_size: float = 0.5,
    sphere_radius: float = 0.05,
    arrow_dims: Tuple[float, float, float, float] = (0.02, 0.04, 0.15, 0.06),  # cyl_r, cone_r, cyl_h, cone_h
    highlight_arrow_scale: float = 5.0,
    line_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    point_size: Optional[float] = None,                 # None -> adaptive
    z_color_range: Optional[Tuple[float, float]] = None,  # None -> auto (min,max)
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    window_title: str = "Open3D Viewer",
    fit_camera: bool = True,

    # --- Misc ---
    print_summary: bool = True,
    summary: Optional[str] = None,

    # --- Return policy ---
    return_visualizer: bool = False,                    # True -> return vis object, don't run/destroy
):
    """
    Unified Open3D visualizer: renders (A) LiDAR point cloud, (B) full GNSS polyline,
    (C) sampled markers + orientation arrows; supports interactive stepping via callbacks.

    Color modes:
      - "z": hue mapped from z-range (blue→red). If z_color_range is None, auto min/max.
      - "per_scan": per-scan solid colors using per_scan_offsets/colors, else auto-fallback to "z".
    Notes:
      - If voxel_size is set, downsampling is applied and per-scan coloring becomes invalid -> falls back to "z".
      - Callbacks keep the same signatures as provided in the original code.
      - If return_visualizer=True, this function creates the window and geometries but does NOT run/destroy;
        call vis.run(); vis.destroy_window() at the call site.
    """
    _ensure_open3d_installed()

    geometries: List[o3d.geometry.Geometry] = []
    marker_spheres: List[o3d.geometry.TriangleMesh] = []
    marker_arrows: List[o3d.geometry.TriangleMesh] = []
    marker_centers: List[np.ndarray] = []

    if print_summary and summary:
        print(summary)

    # Optional axes
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axes_size), origin=[0.0, 0.0, 0.0])
        geometries.append(axes)

    # ---- Point cloud ----
    pcd_ref: Optional[o3d.geometry.PointCloud] = None
    color_mode_active = "z"  # default/fallback
    if points_xyz is not None and points_xyz.size > 0:
        pts = np.asarray(points_xyz, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Decide color mode (fallback rules)
        if color_mode == "per_scan" and per_scan_offsets and per_scan_colors:
            # Validate offsets & color list
            if len(per_scan_offsets) > 0 and len(per_scan_colors) >= len(per_scan_offsets):
                colors_arr = np.zeros((pts.shape[0], 3), dtype=np.float64)
                for i, start in enumerate(per_scan_offsets):
                    end = per_scan_offsets[i + 1] if i + 1 < len(per_scan_offsets) else pts.shape[0]
                    c = np.asarray(per_scan_colors[i], dtype=np.float64)
                    c = np.clip(c, 0.0, 1.0)
                    if start < end and start >= 0:
                        colors_arr[start:end, :] = c
                pcd.colors = o3d.utility.Vector3dVector(colors_arr)
                color_mode_active = "per_scan"
            else:
                color_mode_active = "z"
        else:
            color_mode_active = "z"

        # z-coloring (default / fallback)
        if color_mode_active == "z":
            if pts.size > 0:
                z = pts[:, 2]
                if z_color_range is None:
                    z_min = float(np.min(z)) if z.size > 0 else 0.0
                    z_max = float(np.max(z)) if z.size > 0 else 1.0
                else:
                    z_min, z_max = float(z_color_range[0]), float(z_color_range[1])
                if z_max > z_min:
                    z_norm = (z - z_min) / (z_max - z_min)
                else:
                    z_norm = np.zeros_like(z)
                hues = (2.0 / 3.0) * (1.0 - z_norm)  # 2/3 (blue) -> 0 (red)
                colors = np.zeros((z.shape[0], 3), dtype=np.float64)
                for i, h in enumerate(hues):
                    r, g, b = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
                    colors[i] = [r, g, b]
                pcd.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(pcd)
        pcd_ref = pcd

    # ---- Full polyline with thin black line ----
    if polyline_points_full is not None and polyline_points_full.shape[0] >= 2:
        P = np.asarray(polyline_points_full, dtype=np.float64)
        lines = [[i, i + 1] for i in range(P.shape[0] - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(P),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
        )
        line_set.colors = o3d.utility.Vector3dVector([list(line_color) for _ in lines])
        geometries.append(line_set)

    # ---- Markers (spheres) + orientation arrows ----
    current_highlight: Optional[int] = int(highlight_marker_index) if highlight_marker_index is not None else None

    M = np.asarray(marker_points, dtype=np.float64)
    has_quats = marker_quats_xyzw is not None and marker_quats_xyzw.shape[0] == M.shape[0]
    for i in range(M.shape[0]):
        center = M[i, :]
        marker_centers.append(center.copy())
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(sphere_radius))
        sphere.translate(center)
        if current_highlight is not None and i == current_highlight:
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
            sphere.scale(float(highlight_arrow_scale), center=center)
        else:
            sphere.paint_uniform_color([0.0, 0.0, 0.0])  # black
        geometries.append(sphere)
        marker_spheres.append(sphere)

        if has_quats:
            cyl_r, cone_r, cyl_h, cone_h = arrow_dims
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=float(cyl_r),
                cone_radius=float(cone_r),
                cylinder_height=float(cyl_h),
                cone_height=float(cone_h),
            )
            # Default arrow points +Z -> align +Z to +X by +90deg about +Y
            R_align = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, np.pi / 2.0, 0.0])
            arrow.rotate(R_align, center=(0.0, 0.0, 0.0))
            q = np.asarray(marker_quats_xyzw[i, :], dtype=np.float64)
            R = _quat_xyzw_to_R(q)
            arrow.rotate(R, center=(0.0, 0.0, 0.0))
            arrow.translate(center)
            if current_highlight is not None and i == current_highlight:
                arrow.paint_uniform_color([1.0, 0.0, 0.0])
                arrow.scale(float(highlight_arrow_scale), center=center)
            else:
                arrow.paint_uniform_color([0.0, 0.0, 0.0])
            geometries.append(arrow)
            marker_arrows.append(arrow)

    # ---- Build visualizer ----
    interactive = (pcd_ref is not None) and (scan_updater is not None or time_offset_updater is not None)
    vis = o3d.visualization.VisualizerWithKeyCallback() if interactive else o3d.visualization.Visualizer()
    vis.create_window(window_name=str(window_title))

    for g in geometries:
        vis.add_geometry(g)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color, dtype=np.float64)

    # Point size (adaptive or fixed)
    if pcd_ref is not None:
        if point_size is None:
            try:
                pts_for_scale = np.asarray(pcd_ref.points)
                if pts_for_scale.size > 0:
                    centroid = np.mean(pts_for_scale, axis=0)
                    dists = np.linalg.norm(pts_for_scale - centroid, axis=1)
                    scale = float(np.percentile(dists, 90) + 1e-6)
                    size = 1.5 * (scale ** 0.5)
                    size = float(np.clip(size, 1.0, 8.0))
                    opt.point_size = size
            except Exception:
                pass
        else:
            opt.point_size = float(point_size)

    # Camera fit to union AABB
    if fit_camera and geometries:
        mins = []
        maxs = []
        for g in geometries:
            try:
                bbox = g.get_axis_aligned_bounding_box()
                mins.append(np.asarray(bbox.get_min_bound()))
                maxs.append(np.asarray(bbox.get_max_bound()))
            except Exception:
                pass
        if mins and maxs:
            mins_arr = np.vstack(mins)
            maxs_arr = np.vstack(maxs)
            bbox_all = o3d.geometry.AxisAlignedBoundingBox(np.min(mins_arr, axis=0), np.max(maxs_arr, axis=0))
            vc = vis.get_view_control()
            if hasattr(vc, "fit_to_geometry"):
                vc.fit_to_geometry(bbox_all)
            else:
                # Fallback for Open3D builds without fit_to_geometry
                c = bbox_all.get_center()
                ext = bbox_all.get_extent()
                diag = float(np.linalg.norm(ext))
                front = np.array([0.0, -1.0, -1.5], dtype=np.float64)
                n = np.linalg.norm(front)
                if n > 0:
                    front = front / n
                up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                vc.set_lookat(c)
                vc.set_front(front.tolist())
                vc.set_up(up.tolist())
                try:
                    zoom = float(np.clip(5.0 / (diag + 1e-6), 0.35, 1.2))
                    vc.set_zoom(zoom)
                except Exception:
                    pass

    # ---- Interactive callbacks (if requested) ----
    if interactive and pcd_ref is not None:
        def _apply_z_colors(points_xyz_np: np.ndarray) -> None:
            if points_xyz_np.size == 0:
                return
            z = points_xyz_np[:, 2]
            if z_color_range is None:
                z_min = float(np.min(z)) if z.size > 0 else 0.0
                z_max = float(np.max(z)) if z.size > 0 else 1.0
            else:
                z_min, z_max = float(z_color_range[0]), float(z_color_range[1])
            if z_max > z_min:
                z_norm = (z - z_min) / (z_max - z_min)
            else:
                z_norm = np.zeros_like(z)
            hues = (2.0 / 3.0) * (1.0 - z_norm)
            colors = np.zeros((z.shape[0], 3), dtype=np.float64)
            for i, h in enumerate(hues):
                r, g, b = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
                colors[i] = [r, g, b]
            pcd_ref.colors = o3d.utility.Vector3dVector(colors)

        def _set_highlight(new_idx: Optional[int]) -> None:
            nonlocal current_highlight
            if new_idx is None:
                return
            if current_highlight is not None and 0 <= current_highlight < len(marker_spheres):
                marker_spheres[current_highlight].paint_uniform_color([0.0, 0.0, 0.0])
                try:
                    marker_spheres[current_highlight].scale(1.0 / float(highlight_arrow_scale),
                                                            center=marker_centers[current_highlight])
                except Exception:
                    pass
                if current_highlight < len(marker_arrows):
                    marker_arrows[current_highlight].paint_uniform_color([0.0, 0.0, 0.0])
                    try:
                        marker_arrows[current_highlight].scale(1.0 / float(highlight_arrow_scale),
                                                              center=marker_centers[current_highlight])
                    except Exception:
                        pass
                vis.update_geometry(marker_spheres[current_highlight])
                if current_highlight < len(marker_arrows):
                    vis.update_geometry(marker_arrows[current_highlight])
            current_highlight = int(new_idx)
            if 0 <= current_highlight < len(marker_spheres):
                marker_spheres[current_highlight].paint_uniform_color([1.0, 0.0, 0.0])
                try:
                    marker_spheres[current_highlight].scale(float(highlight_arrow_scale),
                                                            center=marker_centers[current_highlight])
                except Exception:
                    pass
                if current_highlight < len(marker_arrows):
                    marker_arrows[current_highlight].paint_uniform_color([1.0, 0.0, 0.0])
                    try:
                        marker_arrows[current_highlight].scale(float(highlight_arrow_scale),
                                                              center=marker_centers[current_highlight])
                    except Exception:
                        pass
                vis.update_geometry(marker_spheres[current_highlight])
                if current_highlight < len(marker_arrows):
                    vis.update_geometry(marker_arrows[current_highlight])

        def _update_with_points(new_pts: np.ndarray, new_hi: Optional[int]) -> None:
            if new_pts is None or new_pts.size == 0:
                return
            # Note: interactive에서는 per-scan offset을 알 수 없으므로 z 컬러로 고정
            pcd_ref.points = o3d.utility.Vector3dVector(new_pts)
            _apply_z_colors(new_pts)
            _set_highlight(new_hi)
            vis.update_geometry(pcd_ref)
            vis.poll_events()
            vis.update_renderer()

        step = max(1, int(key_step))

        # RIGHT / LEFT to step scan index
        if scan_updater is not None:
            def _step_scan(d: int) -> None:
                pts, _unused, hi = scan_updater(int(d))
                _update_with_points(pts, hi)
            vis.register_key_callback(262, lambda v: (_step_scan(+step), False)[1])  # RIGHT
            vis.register_key_callback(263, lambda v: (_step_scan(-step), False)[1])  # LEFT
            # Also map '[' and ']' as additional step keys
            vis.register_key_callback(ord(']'), lambda v: (_step_scan(+step), False)[1])
            vis.register_key_callback(ord('['), lambda v: (_step_scan(-step), False)[1])

        # ',' and '.' to adjust time offset (seconds)
        if time_offset_updater is not None:
            def _offset(dt: float) -> None:
                pts_new, _unused, new_hi = time_offset_updater(float(dt))
                _update_with_points(pts_new, new_hi)
            vis.register_key_callback(44, lambda v: (_offset(-float(step)), False)[1])  # ','
            vis.register_key_callback(46, lambda v: (_offset(+float(step)), False)[1])  # '.'

    # Run or return
    if return_visualizer:
        return vis

    # Show controls (short)
    if interactive:
        print("Controls: ←/→ or [/]= scan ±step; ,/. = time offset ±step; Q/Esc to exit.")

    vis.run()
    vis.destroy_window()
    return None
