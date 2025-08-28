#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# BEV utils
from lidar_gnss.bev_viz_utils import (
    load_gps_csv, xy_from_df, compute_ci_arrays, build_norms,
    COV_COLS_DEFAULT, select_stride
)

# LiDAR-GNSS pipeline
from lidar_gnss.io_utils import load_gnss_csv, parse_lidar_directory, load_extrinsics
from lidar_gnss.pose_utils import build_interpolators, resample_poses
from lidar_gnss.accumulate import accumulate_lidar_points

# Optional: PyQtGraph for 3D view (install pyqtgraph)
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5 import QtGui
try:
    from OpenGL.GL import glGetDoublev, glGetIntegerv, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
except Exception:
    glGetDoublev = None; glGetIntegerv = None; GL_MODELVIEW_MATRIX = None; GL_PROJECTION_MATRIX = None; GL_VIEWPORT = None


EXTRINSICS_FIXED_PATH = "/home/jiyong/Jiyong/Humzee/gps_vis_tool/extrinsics.yaml"


# ---------------------- BEV Canvases ----------------------

class BEVClickableCanvas(FigureCanvas):
    clicked = QtCore.pyqtSignal(int)  # which cov idx

    def __init__(self, idx: int, parent=None, width=3.2, height=2.4, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self._idx = idx
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=0.5)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()
        self.setMouseTracking(True)

        # Dark theme for MPL figure/axes
        self._fig_bg = "#1e1f22"
        self._ax_bg = "#2b2d31"
        self.fig.patch.set_facecolor(self._fig_bg)
        self.ax.set_facecolor(self._ax_bg)
        for spine in self.ax.spines.values():
            spine.set_color("w")

        # Colorbar handle to prevent accumulation
        self._cbar = None

    def mousePressEvent(self, event):
        self.clicked.emit(self._idx)
        return super().mousePressEvent(event)

    def draw_scatter(self, x: np.ndarray, y: np.ndarray, c: np.ndarray,
                     title: str, unit: str, cmap: str, norm, point_size: float,
                     start_xy: Optional[Tuple[float, float]] = None,
                     show_axes_labels: bool = False):
        # 1) remove previous colorbar/axes
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        for ax_extra in list(self.fig.axes):
            if ax_extra is not self.ax:
                try:
                    self.fig.delaxes(ax_extra)
                except Exception:
                    pass

        # 2) clear and re-apply theme
        self.ax.clear()
        self.fig.patch.set_facecolor(self._fig_bg)
        self.ax.set_facecolor(self._ax_bg)
        for spine in self.ax.spines.values():
            spine.set_color("w")

        # 3) plot
        cplot = np.clip(c, 1e-12, None)
        sc = self.ax.scatter(
            x, y, c=cplot, s=point_size, cmap=cmap, norm=norm,
            linewidths=0, rasterized=True
        )
        self.ax.plot(x, y, "-", color="w", linewidth=0.6, alpha=0.5)

        if start_xy is not None:
            self.ax.scatter([start_xy[0]], [start_xy[1]],
                            s=120, facecolors="none", edgecolors="r", linewidths=2.0)

        self.ax.set_title(title, fontsize=9, color="w")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", linewidth=0.6, color="#555")

        if show_axes_labels:
            self.ax.set_xlabel("X (m)", color="w")
            self.ax.set_ylabel("Y (m)", color="w")
            self.ax.tick_params(colors="w")
        else:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.tick_params(colors="w")

        cb = self.fig.colorbar(sc, ax=self.ax, fraction=0.046, pad=0.04)
        cb.set_label(f"95% CI ({unit})", fontsize=8, color="w")
        cb.ax.tick_params(colors="w")
        try:
            cb.outline.set_edgecolor("w")
        except Exception:
            pass
        self._cbar = cb

        self.fig.tight_layout(pad=0.5)
        self.draw()


class BEVMainCanvas(BEVClickableCanvas):
    def __init__(self, parent=None):
        super().__init__(idx=-1, parent=parent, width=6.8, height=5.6, dpi=100)
        self.setMouseTracking(True)
        self.fig.canvas.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self._hover_annot = None
        self._data_for_hover = None  # (x, y, t_vals, ci_vals, unit, title)
        self._lidar_index_resolver: Optional[Callable[[float], Optional[int]]] = None
        self._install_format_coord()
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_hover_data(self, x: np.ndarray, y: np.ndarray,
                       t_vals: np.ndarray, ci_vals: np.ndarray,
                       unit: str, title: str,
                       lidar_index_resolver: Optional[Callable[[float], Optional[int]]] = None):
        self._data_for_hover = (x, y, t_vals, ci_vals, unit, title)
        self._lidar_index_resolver = lidar_index_resolver
        self._install_format_coord()

    def _install_format_coord(self):
        def fmt(xp, yp):
            if self._data_for_hover is None or xp is None or yp is None:
                try:
                    return f"x={float(xp):.3f}, y={float(yp):.3f}"
                except Exception:
                    return ""
            x_arr, y_arr, t_vals, ci_vals, unit, title = self._data_for_hover
            if x_arr.size == 0:
                try:
                    return f"x={float(xp):.3f}, y={float(yp):.3f}"
                except Exception:
                    return ""
            try:
                i = int(np.argmin((x_arr - float(xp))**2 + (y_arr - float(yp))**2))
                t = float(t_vals[i]); ci = float(ci_vals[i])
                lid_idx_txt = ""
                if self._lidar_index_resolver is not None:
                    li = self._lidar_index_resolver(t)
                    if li is not None:
                        lid_idx_txt = f", lidar_idx={int(li)}"
                return f"{title} | t={t:.4f} s, ci={ci:.3f} {unit}{lid_idx_txt}"
            except Exception:
                try:
                    return f"x={float(xp):.3f}, y={float(yp):.3f}"
                except Exception:
                    return ""
        self.ax.format_coord = fmt

    def reset_hover(self):
        self._hover_annot = None

    def _on_motion(self, event):
        if self._data_for_hover is None or event.inaxes != self.ax:
            return
        x, y, t_vals, ci_vals, unit, title = self._data_for_hover
        if x.size == 0 or event.xdata is None or event.ydata is None:
            return
            cx, cy = float(event.xdata), float(event.ydata)
        i = int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))
        t = float(t_vals[i]); ci = float(ci_vals[i])
        lid_idx_txt = ""
        if self._lidar_index_resolver is not None:
            li = self._lidar_index_resolver(t)
            if li is not None:
                lid_idx_txt = f"\nlidar_idx={int(li)}"
        txt = f"{title}\nt={t:.4f} s\nci={ci:.3f} {unit}{lid_idx_txt}"

        if self._hover_annot is None:
            self._hover_annot = self.ax.annotate(
                txt, xy=(cx, cy), xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc=self._ax_bg, alpha=0.85, ec="w"),
                fontsize=12, color="w"
            )
        else:
            self._hover_annot.set_text(txt)
            self._hover_annot.xy = (cx, cy)
        self.fig.canvas.draw_idle()


# ---------------------- Point Cloud View (PyQtGraph) ----------------------

@dataclass
class CloudState:
    points: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None


class PointCloudView(QtWidgets.QWidget):
    indexChanged = QtCore.pyqtSignal(int)
    offsetChangedMs = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        if gl is None:
            lbl = QtWidgets.QLabel("pyqtgraph not installed. `pip install pyqtgraph`")
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(lbl)
            return

        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view.opts["distance"] = 50
        # Ensure the GL view receives key events and we can intercept them
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)
        self.setFocusProxy(self.view)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)

        # GL items
        self.cloud_item = None
        self.polyline_item = None
        self.start_sphere = None
        self.marker_meshes: List[gl.GLMeshItem] = []
        self.highlight_mesh = None

        # state
        self._has_topdown_set = False
        self._idx_stride = 1
        self._offset_stride_ms = 100
        # continuous motion params
        self._pan_speed_ratio = 1.0   # center shift per second as fraction of distance
        self._zoom_rate_in_per_sec = 0.2  # distance *= 0.65 ** dt
        self._zoom_rate_out_per_sec = 1.0 / self._zoom_rate_in_per_sec
        self._dist_min = 0.05
        self._dist_max = 1e6

        # continuous motion state
        self._pan_dx = 0  # -1,0,1
        self._pan_dy = 0  # -1,0,1
        self._zoom_dir = 0  # -1(out),0,+1(in)
        self._motion_timer = QtCore.QTimer(self)
        self._motion_timer.setInterval(16)  # ~60 FPS
        self._motion_timer.timeout.connect(self._on_motion_tick)
        self._last_motion_ts = time.perf_counter()

        # profiling callback (set by MainWindow)
        self.profile_cb: Optional[Callable[[str], None]] = None

        # marker picking/hover state
        self._marker_points = None
        self._marker_ci_vals = None
        self._marker_times = None
        self._marker_lidar_idx = None
        self._hover_last_idx = None
        self._marker_dirs = None
        self._marker_lengths = None

        # focus for key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _start_motion_timer(self):
        if not self._motion_timer.isActive():
            self._last_motion_ts = time.perf_counter()
            self._motion_timer.start()

    def _stop_motion_timer_if_idle(self):
        if self._pan_dx == 0 and self._pan_dy == 0 and self._zoom_dir == 0:
            self._motion_timer.stop()

    def _apply_zoom_step(self, dt: float) -> None:
        dist = float(self.view.opts.get('distance', 1.0))
        if self._zoom_dir > 0:
            nd = max(self._dist_min, dist * (self._zoom_rate_in_per_sec ** dt))
        else:
            nd = min(self._dist_max, dist * (self._zoom_rate_out_per_sec ** dt))
        self.view.opts['distance'] = nd
        self.view.update()

    def _on_motion_tick(self):
        now = time.perf_counter()
        dt = now - self._last_motion_ts
        self._last_motion_ts = now
        moved = False
        # pan in camera plane
        if self._pan_dx != 0 or self._pan_dy != 0:
            dist = float(self.view.opts.get('distance', 1.0))
            step = float(self._pan_speed_ratio) * max(1e-6, dist) * float(dt)
            r, u = self._compute_right_up_vectors()
            # Ensure left/right and up/down feel correct at pure BEV (near 90° elevation)
            try:
                az = float(self.view.opts.get('azimuth', 0.0))
                el = float(self.view.opts.get('elevation', 0.0))
            except Exception:
                az, el = 0.0, 0.0
            rad = np.pi / 180.0
            near_topdown = abs(np.cos(el * rad)) < 1e-3
            dx_eff = -self._pan_dx if near_topdown else self._pan_dx
            dy_eff = -self._pan_dy if near_topdown else self._pan_dy
            delta = dx_eff * step * r + dy_eff * step * u
            c = self.view.opts.get('center', pg.Vector(0, 0, 0))
            try:
                newc = pg.Vector(float(c.x()) + float(delta[0]), float(c.y()) + float(delta[1]), float(c.z()) + float(delta[2]))
            except Exception:
                newc = pg.Vector(float(c[0]) + float(delta[0]), float(c[1]) + float(delta[1]), float(c[2]) + float(delta[2]))
            self.view.opts['center'] = newc
            moved = True
        # zoom (ratio-based per dt)
        if self._zoom_dir != 0:
            self._apply_zoom_step(dt)
            moved = True
        if moved:
            self.view.update()
        else:
            self._stop_motion_timer_if_idle()

    def _compute_right_up_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        az = float(self.view.opts.get('azimuth', 0.0))
        el = float(self.view.opts.get('elevation', 0.0))
        rad = np.pi / 180.0
        azr = az * rad
        elr = el * rad
        # camera forward (from spherical angles)
        f = np.array([
            np.cos(elr) * np.cos(azr),
            np.cos(elr) * np.sin(azr),
            np.sin(elr)
        ], dtype=np.float64)
        # camera right and up in world
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        r = np.cross(f, up_world)
        n = np.linalg.norm(r)
        # Treat near-top-down as degenerate as well to keep pan camera-relative.
        near_topdown = abs(np.cos(elr)) < 1e-3
        if n < 1e-9 or near_topdown:
            # Use azimuth to define a camera-right in the XY plane so pan stays camera-relative.
            r = np.array([-np.sin(azr), np.cos(azr), 0.0], dtype=np.float64)
            if self.profile_cb is not None:
                try:
                    self.profile_cb(f"right-up near-topdown; using az-based right | az={az:.2f}, el={el:.2f}")
                except Exception:
                    pass
        else:
            r = r / n
        u = np.cross(r, f)
        u = u / (np.linalg.norm(u) + 1e-12)
        return r, u

    def _project_points_to_screen(self, pts: np.ndarray) -> Optional[np.ndarray]:
        # Returns (N,2) window coords or None if unavailable
        try:
            if glGetDoublev is None or glGetIntegerv is None:
                return None
            # ensure GL context
            try:
                self.view.makeCurrent()
            except Exception:
                pass
            mv = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).reshape(4, 4).T
            pr = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).reshape(4, 4).T
            vp = np.array(glGetIntegerv(GL_VIEWPORT), dtype=np.int32)
            M = (mv @ pr)  # Note: order depends on OpenGL; we'll apply as v * M
            # Convert points to clip and then to window
            P = np.concatenate([pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
            clip = P @ M  # row vectors
            w = clip[:, 3:4]
            w[w == 0] = 1e-12
            ndc = clip[:, 0:3] / w
            win = np.zeros((pts.shape[0], 2), dtype=np.float64)
            win[:, 0] = vp[0] + (ndc[:, 0] + 1.0) * 0.5 * vp[2]
            win[:, 1] = vp[1] + (ndc[:, 1] + 1.0) * 0.5 * vp[3]
            return win
        except Exception:
            return None

    def eventFilter(self, obj, ev):
        if obj is self.view:
            if ev.type() == QtCore.QEvent.MouseMove:
                if self._marker_points is not None and self._marker_points.size > 0:
                    pos = ev.pos()
                    win = self._project_points_to_screen(self._marker_points)
                    if win is None:
                        if self.profile_cb:
                            self.profile_cb("hover: projection unavailable (glGet*)")
                        return False
                    if win is not None:
                        dx = win[:, 0] - float(pos.x())
                        dy = (self.view.height() - win[:, 1]) - float(pos.y())  # align origins
                        d2 = dx * dx + dy * dy
                        i = int(np.argmin(d2)) if d2.size else None
                        if i is not None and d2[i] < (12.0 ** 2):
                            if i != self._hover_last_idx:
                                self._hover_last_idx = i
                                # build tooltip
                                ci_txt = ""
                                if self._marker_ci_vals is not None:
                                    try:
                                        ci_txt = f"ci={float(self._marker_ci_vals[i]):.3f}"
                                    except Exception:
                                        ci_txt = ""
                                idx_txt = ""
                                if self._marker_lidar_idx is not None:
                                    try:
                                        idx_txt = f"lidar_idx={int(self._marker_lidar_idx[i])}"
                                    except Exception:
                                        idx_txt = ""
                                t_txt = ""
                                if self._marker_times is not None:
                                    try:
                                        t_txt = f"t={float(self._marker_times[i]):.4f} s"
                                    except Exception:
                                        t_txt = ""
                                parts = [p for p in [t_txt, ci_txt, idx_txt] if p]
                                if parts:
                                    if self.profile_cb:
                                        self.profile_cb(f"hover hit i={i}, d={np.sqrt(d2[i]):.1f}px | {' | '.join(parts)}")
                                    QtWidgets.QToolTip.showText(ev.globalPos(), "\n".join(parts), self.view)
                        else:
                            self._hover_last_idx = None
                return False
            if ev.type() == QtCore.QEvent.MouseButtonPress:
                if self._marker_points is not None and self._marker_points.size > 0:
                    pos = ev.pos()
                    win = self._project_points_to_screen(self._marker_points)
                    if win is None:
                        if self.profile_cb:
                            self.profile_cb("click: projection unavailable (glGet*)")
                        return False
                    if win is not None:
                        dx = win[:, 0] - float(pos.x())
                        dy = (self.view.height() - win[:, 1]) - float(pos.y())
                        d2 = dx * dx + dy * dy
                        i = int(np.argmin(d2)) if d2.size else None
                        if self.profile_cb and i is not None:
                            self.profile_cb(f"click nearest i={i}, d={np.sqrt(d2[i]):.1f}px")
                        if i is not None and d2[i] < (12.0 ** 2):
                            if self._marker_lidar_idx is not None:
                                try:
                                    if self.profile_cb:
                                        self.profile_cb(f"click jump to lidar_idx={int(self._marker_lidar_idx[i])}")
                                    self.indexChanged.emit(int(self._marker_lidar_idx[i]))
                                    ev.accept(); return True
                                except Exception:
                                    pass
            if ev.type() == QtCore.QEvent.KeyPress:
                key = ev.key(); mods = ev.modifiers()
                # Shift + arrows => start/adjust pan
                if mods & Qt.ShiftModifier and key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                    self._pan_dx = 1 if key == Qt.Key_Left else (-1 if key == Qt.Key_Right else self._pan_dx)
                    self._pan_dy = 1 if key == Qt.Key_Up else (-1 if key == Qt.Key_Down else self._pan_dy)
                    self._start_motion_timer()
                    ev.accept(); return True
                # Ctrl + arrows => start/adjust zoom (apply immediate step for responsiveness)
                if mods & Qt.ControlModifier and key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                    self._zoom_dir = 1 if key in (Qt.Key_Up, Qt.Key_Right) else -1
                    # immediate multiplicative step (assume ~1/60 s)
                    self._apply_zoom_step(1.0/60.0)
                    self.view.update()
                    self._start_motion_timer()
                    ev.accept(); return True
                # Our existing custom keys (a/d/q/e)
                if key == Qt.Key_A:
                    self.indexChanged.emit(-self._idx_stride); ev.accept(); return True
                elif key == Qt.Key_D:
                    self.indexChanged.emit(+self._idx_stride); ev.accept(); return True
                elif key == Qt.Key_Q:
                    self.offsetChangedMs.emit(-self._offset_stride_ms); ev.accept(); return True
                elif key == Qt.Key_E:
                    self.offsetChangedMs.emit(+self._offset_stride_ms); ev.accept(); return True
            elif ev.type() == QtCore.QEvent.KeyRelease:
                key = ev.key(); mods = ev.modifiers()
                if key in (Qt.Key_Left, Qt.Key_Right):
                    if self._pan_dx != 0:
                        self._pan_dx = 0
                if key in (Qt.Key_Up, Qt.Key_Down):
                    if self._pan_dy != 0:
                        self._pan_dy = 0
                if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                    self._zoom_dir = 0
                self._stop_motion_timer_if_idle()
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, ev):
        key = ev.key(); mods = ev.modifiers()
        # also handle here if eventFilter didn't
        if mods & Qt.ShiftModifier and key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._pan_dx = 1 if key == Qt.Key_Left else (-1 if key == Qt.Key_Right else self._pan_dx)
            self._pan_dy = 1 if key == Qt.Key_Up else (-1 if key == Qt.Key_Down else self._pan_dy)
            self._start_motion_timer()
            ev.accept(); return
        if mods & Qt.ControlModifier and key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._zoom_dir = 1 if key in (Qt.Key_Up, Qt.Key_Right) else -1
            self._apply_zoom_step(1.0/60.0)
            self.view.update()
            self._start_motion_timer()
            ev.accept(); return
        if key == Qt.Key_A:
            self.indexChanged.emit(-self._idx_stride); ev.accept(); return
        elif key == Qt.Key_D:
            self.indexChanged.emit(+self._idx_stride); ev.accept(); return
        elif key == Qt.Key_Q:
            self.offsetChangedMs.emit(-self._offset_stride_ms); ev.accept(); return
        elif key == Qt.Key_E:
            self.offsetChangedMs.emit(+self._offset_stride_ms); ev.accept(); return
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key_Left, Qt.Key_Right):
            self._pan_dx = 0
        if key in (Qt.Key_Up, Qt.Key_Down):
            self._pan_dy = 0
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._zoom_dir = 0
        self._stop_motion_timer_if_idle()
        super().keyReleaseEvent(ev)

    def set_strides(self, idx_stride: int, offset_stride_ms: int):
        self._idx_stride = max(1, int(idx_stride))
        self._offset_stride_ms = int(offset_stride_ms)

    def set_topdown_camera(self):
        # top-down (look from +Z towards -Z)
        self.view.setCameraPosition(elevation=90, azimuth=0)
        self._has_topdown_set = True

    def clear_items(self):
        if gl is None:
            return
        if self.cloud_item is not None:
            self.view.removeItem(self.cloud_item)
            self.cloud_item = None
        if self.polyline_item is not None:
            self.view.removeItem(self.polyline_item)
            self.polyline_item = None
        if self.start_sphere is not None:
            self.view.removeItem(self.start_sphere)
            self.start_sphere = None
        for it in self.marker_meshes:
            self.view.removeItem(it)
        self.marker_meshes.clear()
        if self.highlight_mesh is not None:
            self.view.removeItem(self.highlight_mesh)
            self.highlight_mesh = None

    def _make_sphere(self, center: np.ndarray, radius: float = 0.3, color=(1.0, 0.0, 0.0, 1.0)):
        md = gl.MeshData.sphere(rows=10, cols=20, radius=radius)
        mi = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="shaded")
        mi.translate(center[0], center[1], center[2])
        return mi

    def _make_cone(self, base: np.ndarray, direction: np.ndarray,
               length: float = 0.8, radius: float = 0.06,
               color=(0.0, 0.0, 0.0, 1.0)) -> gl.GLMeshItem:
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n == 0:
            d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            d = d / n

        # 기본 콘은 +Z 방향으로 길이 length
        md = gl.MeshData.cylinder(rows=1, cols=24, radius=[radius, 0.0], length=length)
        mi = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="shaded")

        # +Z -> d 로 회전
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(z, d)
        axis_n = np.linalg.norm(axis)
        if axis_n > 1e-12:
            axis = axis / axis_n
            ang = float(np.degrees(np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))))
            mi.rotate(ang, axis[0], axis[1], axis[2])
        elif np.dot(z, d) < 0.0:
            # 정반대 방향
            mi.rotate(180.0, 1.0, 0.0, 0.0)

        # 기준점(base)에 배치 (콘의 밑면이 base에 오게 함)
        mi.translate(float(base[0]), float(base[1]), float(base[2]))
        return mi

    def set_polyline_and_markers(self, polyline_xyz: np.ndarray,
                                 start_point: Optional[np.ndarray],
                                 marker_points: np.ndarray,
                                 marker_dirs: np.ndarray,
                                 highlight_idx: Optional[int],
                                 marker_colors: Optional[np.ndarray] = None,
                                 marker_ci_vals: Optional[np.ndarray] = None,
                                 marker_times: Optional[np.ndarray] = None,
                                 marker_lidar_indices: Optional[np.ndarray] = None):
        if gl is None:
            return

        # Polyline (밝은 노란색, 시인성 향상)
        if polyline_xyz is not None and polyline_xyz.shape[0] >= 2:
            # 밝은 회색으로 변경
            plt3d = gl.GLLinePlotItem(pos=polyline_xyz, color=(0.80, 0.80, 0.80, 1.0),
                                      width=2.0, antialias=True)
            self.polyline_item = plt3d
            self.view.addItem(plt3d)

        # Start sphere (빨강)
        if start_point is not None:
            sp = self._make_sphere(start_point, radius=0.35, color=(1, 0, 0, 1))
            self.start_sphere = sp
            self.view.addItem(sp)

        # Marker arrows (cones) — 일반 마커: BEV 색상과 동일한 매핑 사용
        self.marker_meshes.clear()
        if marker_points is not None and marker_dirs is not None and marker_points.shape[0] == marker_dirs.shape[0]:
            for i in range(marker_points.shape[0]):
                p = marker_points[i]
                d = marker_dirs[i]
                col = (0.20, 0.85, 0.95, 1.0)
                if marker_colors is not None and i < marker_colors.shape[0]:
                    c = marker_colors[i]
                    col = (float(c[0]), float(c[1]), float(c[2]), float(c[3]) if c.shape[0] >= 4 else 1.0)
                cone = self._make_cone(p, d, length=0.8, radius=0.06, color=col)
                cone.setGLOptions('opaque')  # 깊이/불투명(기본)
                self.marker_meshes.append(cone)
                self.view.addItem(cone)

        # Highlight arrow — 현재 시점 화살표는 빨간색 고정, additive로 최상단 가시성
        if highlight_idx is not None and 0 <= highlight_idx < marker_points.shape[0]:
            p = marker_points[highlight_idx]
            d = marker_dirs[highlight_idx]
            # 순수한 빨간색
            hcol = (1.0, 0.0, 0.0, 1.0)
            self.highlight_mesh = self._make_cone(p, d, length=1.2, radius=0.08, color=hcol)
            self.highlight_mesh.setGLOptions('additive')  # 블렌딩 우선으로 다른 메쉬 위에 보이게
            self.view.addItem(self.highlight_mesh)

        # store marker metadata for picking/hover
        self._marker_points = marker_points.copy() if marker_points is not None else None
        self._marker_ci_vals = marker_ci_vals.copy() if marker_ci_vals is not None else None
        self._marker_times = marker_times.copy() if marker_times is not None else None
        self._marker_lidar_idx = marker_lidar_indices.copy() if marker_lidar_indices is not None else None
        self._hover_last_idx = None
        if self.profile_cb:
            try:
                n = 0 if self._marker_points is None else self._marker_points.shape[0]
                self.profile_cb(f"markers set | N={n}, has_ci={self._marker_ci_vals is not None}, has_times={self._marker_times is not None}, has_lidar_idx={self._marker_lidar_idx is not None}")
            except Exception:
                pass

        if not self._has_topdown_set:
            self.set_topdown_camera()

    def set_pointcloud(self, xyz: np.ndarray):
        if gl is None:
            return
        if xyz is None or xyz.size == 0:
            return
        t0 = time.perf_counter()
        z = xyz[:, 2]
        zmin = float(np.min(z)) if z.size else 0.0
        zmax = float(np.max(z)) if z.size else 1.0
        if zmax > zmin:
            zn = (z - zmin) / (zmax - zmin)
        else:
            zn = np.zeros_like(z)
        # Vectorized HSV->RGB
        hues = (2.0 / 3.0) * (1.0 - zn)
        hsv = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
        rgb = mcolors.hsv_to_rgb(hsv).astype(np.float32)
        colors = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        if self.profile_cb:
            self.profile_cb(f"color mapping: {(time.perf_counter()-t0)*1000:.1f} ms for {xyz.shape[0]} pts")

        # 아주 작게: 픽셀 모드 + 작은 사이즈
        t1 = time.perf_counter()
        sp = gl.GLScatterPlotItem(pos=xyz.astype(np.float32), color=colors, size=1.0, pxMode=True)
        if self.cloud_item is not None:
            self.view.removeItem(self.cloud_item)
        self.cloud_item = sp
        self.view.addItem(sp)
        if self.profile_cb:
            self.profile_cb(f"GL scatter upload: {(time.perf_counter()-t1)*1000:.1f} ms")


# ---------------------- Options Dialog ----------------------

class OptionsDialog(QtWidgets.QDialog):
    reloadRequested = QtCore.pyqtSignal()
    def __init__(self, parent=None, *, offset_min_ms=-1000, offset_max_ms=1000,
                 offset_slider_step_ms=1, marker_stride=5,
                 range_enabled=False, x_range=30.0, y_range=30.0, z_range=30.0,
                 max_points=0, max_frames=10, index_interval=1):
        super().__init__(parent)
        self.setWindowTitle("Options")
        # Dark theme for dialog
        self.setStyleSheet("""
            QDialog { background-color: #1e1f22; color: #e6e6e6; }
            QLabel { color: #e6e6e6; }
            QGroupBox { border: 1px solid #3b3f45; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; color: #e6e6e6; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1f2023; color: #e6e6e6; border: 1px solid #3b3f45;
            }
            QCheckBox { color: #e6e6e6; }
            QDialogButtonBox { background-color: #2b2d31; }
            QPushButton { background-color: #2b2d31; color: #e6e6e6; border: 1px solid #3b3f45; padding: 4px 8px; }
            QPushButton:hover { background-color: #3b3f45; }
        """)

        # Time offset section
        self.offsetMin = QtWidgets.QDoubleSpinBox(); self.offsetMin.setRange(-1e6, 0); self.offsetMin.setDecimals(1); self.offsetMin.setValue(offset_min_ms)
        self.offsetMax = QtWidgets.QDoubleSpinBox(); self.offsetMax.setRange(0, 1e6); self.offsetMax.setDecimals(1); self.offsetMax.setValue(offset_max_ms)
        self.offsetStep = QtWidgets.QDoubleSpinBox(); self.offsetStep.setRange(0.1, 1000.0); self.offsetStep.setDecimals(1); self.offsetStep.setSingleStep(0.1); self.offsetStep.setValue(offset_slider_step_ms)

        # Markers
        self.markerStride = QtWidgets.QSpinBox(); self.markerStride.setRange(1, 1000); self.markerStride.setValue(marker_stride)

        # Lidar merging
        self.maxPoints = QtWidgets.QSpinBox(); self.maxPoints.setRange(0, 1000000000); self.maxPoints.setValue(int(max_points))
        self.maxFrames = QtWidgets.QSpinBox(); self.maxFrames.setRange(1, 100000); self.maxFrames.setValue(int(max_frames))
        self.indexInterval = QtWidgets.QSpinBox(); self.indexInterval.setRange(1, 1000); self.indexInterval.setValue(int(index_interval))

        # Range filter
        self.rangeEnable = QtWidgets.QCheckBox("Enable XYZ range filter"); self.rangeEnable.setChecked(bool(range_enabled))
        self.xRange = QtWidgets.QDoubleSpinBox(); self.xRange.setRange(0.1, 1000.0); self.xRange.setValue(x_range)
        self.yRange = QtWidgets.QDoubleSpinBox(); self.yRange.setRange(0.1, 1000.0); self.yRange.setValue(y_range)
        self.zRange = QtWidgets.QDoubleSpinBox(); self.zRange.setRange(0.1, 1000.0); self.zRange.setValue(z_range)

        form = QtWidgets.QFormLayout()
        form.addRow("Time offset min (ms)", self.offsetMin)
        form.addRow("Time offset max (ms)", self.offsetMax)
        form.addRow("Time offset slider step (ms)", self.offsetStep)
        form.addRow("Arrow marker stride", self.markerStride)
        form.addRow(QtWidgets.QLabel("LiDAR merging"))
        form.addRow("max_points (0=disabled)", self.maxPoints)
        form.addRow("max_frames", self.maxFrames)
        form.addRow("index_interval", self.indexInterval)
        form.addRow(self.rangeEnable)
        form.addRow("x range (m)", self.xRange)
        form.addRow("y range (m)", self.yRange)
        form.addRow("z range (m)", self.zRange)

        btnBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btnBox.accepted.connect(self.accept); btnBox.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(btnBox)

    def values(self):
        return dict(
            offset_min_ms=float(self.offsetMin.value()),
            offset_max_ms=float(self.offsetMax.value()),
            offset_slider_step_ms=float(self.offsetStep.value()),
            marker_stride=int(self.markerStride.value()),
            max_points=int(self.maxPoints.value()),
            max_frames=int(self.maxFrames.value()),
            index_interval=int(self.indexInterval.value()),
            range_enabled=bool(self.rangeEnable.isChecked()),
            x_range=float(self.xRange.value()), y_range=float(self.yRange.value()), z_range=float(self.zRange.value()),
        )


# ---------------------- Main Window ----------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR-GNSS Visualizer (PyQt)")

        # Data/state
        self.df_gps: Optional[pd.DataFrame] = None
        self.scans = None
        self._lidar_times: Optional[np.ndarray] = None
        self._scan_file_indices: Optional[np.ndarray] = None
        self.extrinsics = None
        self.interps = None
        self.resampled = None
        self.polyline_full = None
        self.origin_shift = np.zeros(3, dtype=np.float64)
        
        # profiling toggle
        self._prof_enabled = False  # set False to silence profiling prints
        self._t0_main = 0.0

        # BEV state
        self.cov_cols = COV_COLS_DEFAULT
        self.cmap = "viridis"
        self.point_size = 6.0
        self.norm_mode = "per"
        self._bev_stride = 1
        self.selected_cov_idx = 0
        self._ci_all_full: Optional[List[np.ndarray]] = None
        self._bev_t_full: Optional[np.ndarray] = None

        # Controls state
        self.offset_min_ms = -1000.0
        self.offset_max_ms = 1000.0
        self.offset_slider_step_ms = 1.0
        self.offset_ms = 0.0
        self.offset_stride_ms = 100  # q/e default
        self.index_stride = 1
        self.current_index = 0

        # Filter/markers
        self.marker_stride = 5
        self.range_enabled = False
        self.x_range = np.inf
        self.y_range = np.inf
        self.z_range = np.inf

        self.max_points = 0           # 0 -> disabled
        self.max_frames = 10
        self.index_interval = 1

        self._build_ui()

    def _pstart(self):
        if self._prof_enabled:
            self._t0_main = time.perf_counter()

    def _plog(self, msg: str):
        if self._prof_enabled:
            dt = (time.perf_counter() - self._t0_main) * 1000.0
            print(f"[prof] {dt:8.1f} ms | {msg}")

    # ---- UI ----
    def _build_ui(self):
        toolbar = QtWidgets.QToolBar("Main")
        self.addToolBar(toolbar)

        # 기존 open 액션들
        actOpenGps = QtWidgets.QAction("Open GPS CSV", self)
        actOpenLidar = QtWidgets.QAction("Open LiDAR Dir", self)
        actOpenFolder = QtWidgets.QAction("Open Folder", self)

        actOptions = QtWidgets.QAction("Options", self)
        actViewCloud = QtWidgets.QAction("View Point Cloud", self)
        actViewCloud.setCheckable(True)
        actViewCloud.setChecked(False)

        # Open 메뉴 버튼
        openBtn = QtWidgets.QToolButton()
        openBtn.setText("Open")
        openBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        openMenu = QtWidgets.QMenu(openBtn)
        openMenu.addAction(actOpenFolder)
        openMenu.addAction(actOpenGps)
        openMenu.addAction(actOpenLidar)
        openBtn.setMenu(openMenu)

        toolbar.addWidget(openBtn)
        toolbar.addSeparator()
        toolbar.addAction(actOptions)

        # 시그널
        actOpenGps.triggered.connect(self._on_open_gps)
        actOpenLidar.triggered.connect(self._on_open_lidar)
        actOpenFolder.triggered.connect(self._on_open_folder)
        actOptions.triggered.connect(self._on_options)
        # actViewCloud.toggled.connect(self._on_toggle_cloud) # 툴바에서 제거

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Center stack: BEV main plot vs PointCloud view
        self.centerStack = QtWidgets.QStackedWidget()
        # BEV main panel
        bevPanel = QtWidgets.QWidget()
        bevV = QtWidgets.QVBoxLayout(bevPanel); bevV.setContentsMargins(6,6,6,6); bevV.setSpacing(4)
        self.main_canvas = BEVMainCanvas(bevPanel)
        self.main_toolbar = NavToolbar(self.main_canvas, bevPanel)
        bevV.addWidget(self.main_toolbar)
        bevV.addWidget(self.main_canvas, 1)
        self.centerStack.addWidget(bevPanel)

        # Point cloud view
        self.pcView = PointCloudView()
        # connections for key-based updates
        self.pcView.indexChanged.connect(self._on_index_step)
        self.pcView.offsetChangedMs.connect(self._on_offset_step_ms)
        # wire profiling callback
        self.pcView.profile_cb = self._plog
        self.centerStack.addWidget(self.pcView)

        # Right panel: thumbnails + controls (vertical, single column)
        rightPanel = QtWidgets.QWidget()
        rightV = QtWidgets.QVBoxLayout(rightPanel); rightV.setContentsMargins(6,6,6,6); rightV.setSpacing(8)

        self.thumb_container = QtWidgets.QWidget()
        self.thumb_layout = QtWidgets.QVBoxLayout(self.thumb_container)
        self.thumb_layout.setContentsMargins(0,0,0,0)
        self.thumb_layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.thumb_container)

        rightV.addWidget(scroll, 1)

        # Controls group
        ctrlGroup = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrlGroup)

        # Metric selection (sync with thumbnails/main BEV)
        self.metricCombo = QtWidgets.QComboBox()
        self.metricCombo.setEnabled(False)
        self.metricCombo.currentIndexChanged.connect(self._on_metric_changed)
        form.addRow("Metric", self.metricCombo)

        # Time offset: slider + spin + stride input
        self.offsetSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.offsetSlider.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSlider.setSingleStep(int(self.offset_slider_step_ms))
        self.offsetSlider.setValue(int(self.offset_ms))
        # Time offset spinbox: 엔터로만 반영
        self.offsetSpin = QtWidgets.QSpinBox()
        self.offsetSpin.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSpin.setSingleStep(1)
        self.offsetSpin.setSuffix(" ms")
        self.offsetSpin.setValue(int(self.offset_ms))
        self.offsetSpin.setKeyboardTracking(False)  # 타이핑 중엔 valueChanged 안 나가게
        self.offsetSpin.lineEdit().returnPressed.connect(self._on_offset_spin_commit)

        self.offsetSlider.valueChanged.connect(self._on_offset_slider)
        self.offsetStrideSpin = QtWidgets.QSpinBox()
        self.offsetStrideSpin.setRange(1, 100000)
        self.offsetStrideSpin.setValue(int(self.offset_stride_ms))
        self.offsetStrideSpin.setSuffix(" ms (q/e)")

        self.offsetSlider.valueChanged.connect(self._on_offset_slider)
        self.offsetSpin.valueChanged.connect(self._on_offset_spin)
        self.offsetStrideSpin.valueChanged.connect(self._on_offset_stride_changed)

        offRow = QtWidgets.QHBoxLayout()
        offRow.addWidget(self.offsetSlider, 1)
        offRow.addWidget(self.offsetSpin, 0)
        form.addRow("Time offset (ms)", offRow)
        form.addRow("Offset stride", self.offsetStrideSpin)

        # LiDAR index: slider + spin + stride input
        self.indexSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.indexSlider.setRange(0, 0)
        self.indexSlider.setValue(0)
        # LiDAR index spinbox: 엔터로만 반영
        self.indexSpin = QtWidgets.QSpinBox()
        self.indexSpin.setRange(0, 0)
        self.indexSpin.setValue(0)
        self.indexSpin.setKeyboardTracking(False)
        self.indexSpin.lineEdit().returnPressed.connect(self._on_index_spin_commit)

        # Missing: create index stride spinbox before use
        self.indexStrideSpin = QtWidgets.QSpinBox()
        self.indexStrideSpin.setRange(1, 10000)
        self.indexStrideSpin.setValue(self.index_stride)
        self.indexStrideSpin.setSuffix(" (a/d)")

        self.indexSlider.valueChanged.connect(self._on_index_slider)
        self.indexSpin.valueChanged.connect(self._on_index_spin)
        self.indexStrideSpin.valueChanged.connect(self._on_index_stride_changed)

        idxRow = QtWidgets.QHBoxLayout()
        idxRow.addWidget(self.indexSlider, 1)
        idxRow.addWidget(self.indexSpin, 0)
        form.addRow("LiDAR index", idxRow)
        form.addRow("Index stride", self.indexStrideSpin)

        # View Point Cloud toggle button
        self.viewCloudToggle = QtWidgets.QPushButton("View Point Cloud")
        self.viewCloudToggle.setCheckable(True)
        self.viewCloudToggle.setChecked(False)
        self.viewCloudToggle.toggled.connect(self._on_toggle_cloud)
        rightV.addWidget(self.viewCloudToggle, 0)

        # (Removed) Reload button from panel — moved into Options dialog

        rightV.addWidget(ctrlGroup, 0)

        splitter.addWidget(self.centerStack)
        splitter.addWidget(rightPanel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 380])

        # stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1f22; color: #e6e6e6; }
            QWidget { color: #e6e6e6; }

            /* Top toolbars (including Matplotlib toolbar) */
            QToolBar {
                background-color: #2b2d31;
                border: 0px;
                spacing: 6px;
                padding: 4px;
            }
            QToolBar QToolButton {
                color: #e6e6e6;
                background: transparent;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #3b3f45;
            }
            QToolBar QToolButton:pressed, QToolBar QToolButton:checked {
                background-color: #4b4f55;
            }

            /* Groups, labels, inputs */
            QGroupBox { border: 1px solid #3b3f45; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; color: #e6e6e6; }
            QLabel { color: #e6e6e6; }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #1f2023;
                color: #e6e6e6;
                border: 1px solid #3b3f45;
            }
            QComboBox {
                color: #000000; /* 글자색 검정 */
                background-color: #e6e6e6;
                border: 1px solid #3b3f45;
                padding: 2px 6px;
            }
            QComboBox QAbstractItemView {
                color: #000000;
                background-color: #e6e6e6;
                selection-background-color: #cfcfcf;
                selection-color: #000000;
            }

            /* Sliders */
            QSlider::groove:horizontal { background: #3b3f45; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e6e6e6; width: 12px; margin: -4px 0; border-radius: 6px; }

            /* Scroll areas and status */
            QScrollArea, QAbstractScrollArea { background-color: #1e1f22; }
            QStatusBar { background-color: #2b2d31; color: #e6e6e6; }
            QMenu {
                background-color: #2b2d31;
                color: #e6e6e6;
                border: 1px solid #3b3f45;
            }
            QMenu::item:selected {
                background-color: #3b3f45;
            }
            QPushButton {
                background-color: #2b2d31;
                color: #e6e6e6;
                border: 1px solid #3b3f45;
                padding: 6px 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3b3f45; }
            QPushButton:pressed, QPushButton:checked { background-color: #4b4f55; }
        """)

        # Global shortcuts as fallback (active only on point cloud view)
        self.shortA = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self); self.shortA.setContext(Qt.ApplicationShortcut)
        self.shortD = QtWidgets.QShortcut(QtGui.QKeySequence("D"), self); self.shortD.setContext(Qt.ApplicationShortcut)
        self.shortQ = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self); self.shortQ.setContext(Qt.ApplicationShortcut)
        self.shortE = QtWidgets.QShortcut(QtGui.QKeySequence("E"), self); self.shortE.setContext(Qt.ApplicationShortcut)

        def _shortcut_guarded(callable_):
            def inner():
                if self.centerStack.currentIndex() == 1:
                    callable_()
            return inner

        self.shortA.activated.connect(_shortcut_guarded(lambda: self._on_index_step(-self.index_stride)))
        self.shortD.activated.connect(_shortcut_guarded(lambda: self._on_index_step(+self.index_stride)))
        self.shortQ.activated.connect(_shortcut_guarded(lambda: self._on_offset_step_ms(-self.offset_stride_ms)))
        self.shortE.activated.connect(_shortcut_guarded(lambda: self._on_offset_step_ms(+self.offset_stride_ms)))

    # ---- Toolbar actions ----
    def _on_open_gps(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open GPS CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            # PointCloud pipeline용 df (pos_x/ori_* 필수)
            self.df_gps = load_gnss_csv(path, verbose=False)
        except Exception:
            # BEV만이라도 보이게 로버스트 로더
            self.df_gps = load_gps_csv(path)

        self._prepare_gps_dependent()

    def _on_open_lidar(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open LiDAR Directory", "")
        if not d:
            return
        self.scans = parse_lidar_directory(d)
        self._lidar_times = np.array([s.t for s in self.scans], dtype=np.float64)
        self._refresh_index_bounds()

        # extrinsics, interps, resampled, polyline 준비 (GPS가 있어야 함)
        if self.df_gps is not None:
            self._prepare_pointcloud_support()

    def _auto_find_in_folder(self, root_dir: str) -> Tuple[Optional[str], Optional[str]]:
        # 1) 고정 경로 우선
        candidates_csv = [
            os.path.join(root_dir, "GPS", "Odom_data.csv"),
            os.path.join(root_dir, "GPS_data.csv"),
            os.path.join(root_dir, "Odom_data.csv"),
        ]
        csv_path = next((p for p in candidates_csv if os.path.isfile(p)), None)

        # 재귀적으로 보조 탐색 (이름 유사)
        if csv_path is None:
            for dirpath, _dirnames, filenames in os.walk(root_dir):
                for name in filenames:
                    lower = name.lower()
                    if lower in ("odom_data.csv", "gps_data.csv"):
                        csv_path = os.path.join(dirpath, name)
                        break
                if csv_path:
                    break

        # 2) LiDAR 폴더 찾기: xyzi.bin 존재 여부 또는 파일명 패턴
        lidar_dir = None
        # 우선 고정 이름
        if os.path.isdir(os.path.join(root_dir, "lidar_xyzi")):
            lidar_dir = os.path.join(root_dir, "lidar_xyzi")
        else:
            # 가장 많은 'xyzi.bin'을 가진 폴더 선택
            best_dir, best_count = None, 0
            for dirpath, _dirnames, filenames in os.walk(root_dir):
                count = sum(1 for f in filenames if f.endswith("xyzi.bin"))
                if count > best_count:
                    best_count, best_dir = count, dirpath
            if best_count > 0:
                lidar_dir = best_dir

        return csv_path, lidar_dir

    def _on_open_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if not d:
            return
        csv_path, lidar_dir = self._auto_find_in_folder(d)

        # GPS
        if csv_path:
            try:
                self.df_gps = load_gnss_csv(csv_path, verbose=False)
            except Exception:
                self.df_gps = load_gps_csv(csv_path)
            self._prepare_gps_dependent()
        else:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Could not find GPS CSV in the selected folder.")

        # LiDAR
        if lidar_dir:
            try:
                self.scans = parse_lidar_directory(lidar_dir)
                self._lidar_times = np.array([s.t for s in self.scans], dtype=np.float64)
                self._refresh_index_bounds()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "LiDAR Error", str(e))
        else:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Could not find LiDAR directory in the selected folder.")

        # 포인트클라우드 의존 준비
        if self.df_gps is not None:
            self._prepare_pointcloud_support()

    def _on_options(self):
        dlg = OptionsDialog(
            self,
            offset_min_ms=self.offset_min_ms,
            offset_max_ms=self.offset_max_ms,
            offset_slider_step_ms=self.offset_slider_step_ms,
            marker_stride=self.marker_stride,
            range_enabled=self.range_enabled,
            x_range=self.x_range if np.isfinite(self.x_range) else 30.0,
            y_range=self.y_range if np.isfinite(self.y_range) else 30.0,
            z_range=self.z_range if np.isfinite(self.z_range) else 30.0,
            max_points=self.max_points,
            max_frames=self.max_frames,
            index_interval=self.index_interval,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            v = dlg.values()
            self._apply_options_from_values(v)
            # OK 누르면 즉시 포인트클라우드 리로드
            self._update_pointcloud(force=True)

    def _apply_options_from_values(self, v: dict) -> None:
        self.offset_min_ms = float(v["offset_min_ms"]); self.offset_max_ms = float(v["offset_max_ms"])
        self.offset_slider_step_ms = float(v["offset_slider_step_ms"])
        self.marker_stride = int(v["marker_stride"])
        self.max_points = int(v["max_points"])
        self.max_frames = int(v["max_frames"])
        self.index_interval = int(v["index_interval"])
        self.range_enabled = bool(v["range_enabled"])
        if self.range_enabled:
            self.x_range = float(v["x_range"]); self.y_range = float(v["y_range"]); self.z_range = float(v["z_range"]) 
        else:
            self.x_range = self.y_range = self.z_range = np.inf
        # apply to slider widgets
        self.offsetSlider.blockSignals(True)
        self.offsetSpin.blockSignals(True)
        self.offsetSlider.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSpin.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSlider.setSingleStep(int(max(1, round(self.offset_slider_step_ms))))
        self.offsetSlider.blockSignals(False)
        self.offsetSpin.blockSignals(False)

    def _apply_options_and_reload(self, dlg: OptionsDialog) -> None:
        v = dlg.values()
        self._apply_options_from_values(v)
        self._update_pointcloud(force=True)

    def _on_toggle_cloud(self, checked: bool):
        self.centerStack.setCurrentIndex(1 if checked else 0)
        if checked:
            self.pcView.setFocus()
            self.pcView.view.setFocus()   # ensure GL widget has focus
            self.pcView.activateWindow()   # <- 추가
            self.pcView.raise_()           # <- 추가
            self.pcView.set_strides(self.index_stride, int(self.offset_stride_ms))
            # Ensure index slider/spin reflect actual scan range
            self._refresh_index_bounds()
            # 초기 뷰셋업
            self._update_pointcloud(force=True)

    # ---- Control handlers ----
    def _on_offset_slider(self, val: int):
        self.offset_ms = float(val)
        self.offsetSpin.blockSignals(True)
        self.offsetSpin.setValue(int(val))
        self.offsetSpin.blockSignals(False)
        self._update_pointcloud()

    def _on_offset_spin(self, val: int):
        # This handler is now only for the slider's valueChanged signal
        pass

    def _on_offset_spin_commit(self):
        val = int(self.offsetSpin.value())
        if val != int(self.offset_ms):
            self.offset_ms = float(val)
            self.offsetSlider.blockSignals(True)
            self.offsetSlider.setValue(val)
            self.offsetSlider.blockSignals(False)
            self._update_pointcloud()

    def _on_offset_stride_changed(self, val: int):
        self.offset_stride_ms = int(val)
        self.pcView.set_strides(self.index_stride, int(self.offset_stride_ms))

    def _on_index_slider(self, val: int):
        if self.scans is None:
            return
        if self._scan_file_indices is None or self._scan_file_indices.size == 0:
            self.current_index = int(np.clip(val, 0, len(self.scans) - 1))
            self.indexSpin.blockSignals(True)
            self.indexSpin.setValue(int(val))
            self.indexSpin.blockSignals(False)
            self._update_pointcloud()
            return
        arr = self._scan_file_indices
        i = int(np.argmin(np.abs(arr - int(val))))
        self.current_index = i
        snapped = int(arr[i])
        self.indexSpin.blockSignals(True)
        self.indexSpin.setValue(snapped)
        self.indexSpin.blockSignals(False)
        self._update_pointcloud()

    def _on_index_spin(self, val: int):
        # This handler is now only for the slider's valueChanged signal
        pass

    def _on_index_spin_commit(self):
        if self.scans is None:
            return
        val = int(self.indexSpin.value())
        if self._scan_file_indices is None or self._scan_file_indices.size == 0:
            v = int(np.clip(val, 0, len(self.scans) - 1))
            if v != self.current_index:
                self.current_index = v
                self.indexSlider.blockSignals(True)
                self.indexSpin.blockSignals(True)
                self.indexSlider.setValue(v)
                self.indexSpin.setValue(v)
                self.indexSlider.blockSignals(False)
                self.indexSpin.blockSignals(False)
                self._update_pointcloud()
            return
        arr = self._scan_file_indices
        i = int(np.argmin(np.abs(arr - val)))
        if i != self.current_index:
            self.current_index = i
            snapped = int(arr[i])
            self.indexSlider.blockSignals(True)
            self.indexSpin.blockSignals(True)
            self.indexSlider.setValue(snapped)
            self.indexSpin.setValue(snapped)
            self.indexSlider.blockSignals(False)
            self.indexSpin.blockSignals(False)
            self._update_pointcloud()

    def _on_index_stride_changed(self, val: int):
        self.index_stride = max(1, int(val))
        self.pcView.set_strides(self.index_stride, int(self.offset_stride_ms))

    def _on_index_step(self, delta: int):
        if self.scans is None:
            return
        new_idx = int(np.clip(self.current_index + delta, 0, len(self.scans) - 1))
        if new_idx != self.current_index:
            self.current_index = new_idx
            snapped = int(self._scan_file_indices[new_idx]) if (self._scan_file_indices is not None and self._scan_file_indices.size > 0) else new_idx
            self.indexSlider.blockSignals(True)
            self.indexSpin.blockSignals(True)
            self.indexSlider.setValue(snapped)
            self.indexSpin.setValue(snapped)
            self.indexSlider.blockSignals(False)
            self.indexSpin.blockSignals(False)
            self._update_pointcloud()

    def _on_offset_step_ms(self, delta_ms: int):
        new_val = int(np.clip(self.offset_ms + delta_ms, self.offset_min_ms, self.offset_max_ms))
        if new_val != int(self.offset_ms):
            self.offset_ms = float(new_val)
            self.offsetSlider.blockSignals(True)
            self.offsetSpin.blockSignals(True)
            self.offsetSlider.setValue(int(new_val))
            self.offsetSpin.setValue(int(new_val))
            self.offsetSlider.blockSignals(False)
            self.offsetSpin.blockSignals(False)
            self._update_pointcloud()

    # ---- Data prep ----
    def _prepare_gps_dependent(self):
        if self.df_gps is None:
            return
        # BEV data
        x_m, y_m, df2 = xy_from_df(self.df_gps, origin_lat=None, origin_lon=None)
        self._bev_x, self._bev_y = select_stride(x_m, y_m, self._bev_stride)
        self._bev_df = df2.reset_index(drop=True)
        self._bev_t = self._bev_df["t"].to_numpy(dtype=np.float64) if "t" in self._bev_df.columns else None

        ci_all = compute_ci_arrays(self._bev_df, self.cov_cols)
        self._bev_ci_vals = [select_stride(r.ci_values, r.ci_values, self._bev_stride)[0] for r in ci_all]
        self._bev_units = [r.unit for r in ci_all]
        self._bev_norms = build_norms(ci_all, mode=self.norm_mode)
        # Store full CI arrays and times for 3D color mapping
        self._ci_all_full = [np.asarray(r.ci_values, dtype=np.float64) for r in ci_all]
        self._bev_t_full = self._bev_df["t"].to_numpy(dtype=np.float64) if "t" in self._bev_df.columns else None

        # Populate metric combo
        self.metricCombo.blockSignals(True)
        self.metricCombo.clear()
        self.metricCombo.addItems(self.cov_cols)
        self.metricCombo.setEnabled(True)
        self.metricCombo.setCurrentIndex(int(self.selected_cov_idx))
        self.metricCombo.blockSignals(False)

        # thumbnails
        self._build_thumbnails()
        # initialize main
        self._show_bev_idx(0)

    def _build_thumbnails(self):
        # clear
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self.thumb_canvases: List[BEVClickableCanvas] = []
        for i, name in enumerate(self.cov_cols):
            canvas = BEVClickableCanvas(idx=i, parent=self.thumb_container, width=3.2, height=2.4, dpi=100)
            canvas.clicked.connect(self._show_bev_idx)
            self.thumb_canvases.append(canvas)
            self.thumb_layout.addWidget(canvas)

        # render
        start_xy = (self._bev_x[0], self._bev_y[0]) if self._bev_x.size > 0 else None
        for i, canvas in enumerate(self.thumb_canvases):
            canvas.draw_scatter(
                self._bev_x, self._bev_y, self._bev_ci_vals[i],
                title=self.cov_cols[i], unit=self._bev_units[i], cmap=self.cmap,
                norm=self._bev_norms[i], point_size=self.point_size,
                start_xy=start_xy, show_axes_labels=False
            )

        # spacer
        self.thumb_layout.addStretch(1)

    @QtCore.pyqtSlot(int)
    def _show_bev_idx(self, idx: int):
        if idx < 0 or idx >= len(self.cov_cols):
            return
        # 이전 hover 주석 초기화
        self.main_canvas.reset_hover()  # <- 추가
        # sync selected metric
        self.selected_cov_idx = int(idx)
        if hasattr(self, 'metricCombo'):
            self.metricCombo.blockSignals(True)
            self.metricCombo.setCurrentIndex(int(idx))
            self.metricCombo.blockSignals(False)

        start_xy = (self._bev_x[0], self._bev_y[0]) if self._bev_x.size > 0 else None
        self.main_canvas.draw_scatter(
            self._bev_x, self._bev_y, self._bev_ci_vals[idx],
            title=self.cov_cols[idx], unit=self._bev_units[idx],
            cmap=self.cmap, norm=self._bev_norms[idx], point_size=self.point_size,
            start_xy=start_xy, show_axes_labels=True
        )
        # hover payload with LiDAR index resolver
        t_vals = self._bev_t if self._bev_t is not None and len(self._bev_t) == len(self._bev_x) else np.arange(len(self._bev_x), dtype=float)
        self.main_canvas.set_hover_data(
            self._bev_x, self._bev_y, t_vals, self._bev_ci_vals[idx], self._bev_units[idx], self.cov_cols[idx],
            lidar_index_resolver=self._lidar_index_for_gps_time
        )
        # recolor 3D if visible
        if self.centerStack.currentIndex() == 1:
            self._update_pointcloud(force=True)

    def _on_metric_changed(self, idx: int):
        if idx < 0 or idx >= len(self.cov_cols):
            return
        # forward to BEV show and trigger color sync
        self._show_bev_idx(int(idx))

    def _lidar_index_for_gps_time(self, t_gps: float) -> Optional[int]:
        if self._lidar_times is None or self._lidar_times.size == 0:
            return None
        # GPS-based offset: gps_time + offset aligns to (scan.t + lidar_time_offset_s)
        gps_offset_s = float(self.offset_ms) * 1e-3
        lidar_time_offset_s = -gps_offset_s
        adjusted = self._lidar_times + lidar_time_offset_s
        # nearest index
        i = int(np.argmin(np.abs(adjusted - float(t_gps))))
        return i

    def _refresh_index_bounds(self):
        if self.scans is None:
            return
        # build actual LiDAR file indices array
        self._scan_file_indices = np.array([s.index for s in self.scans], dtype=int)
        total = len(self.scans)
        max_idx = max(0, total - 1)
        # clamp current internal array index
        cur = int(np.clip(self.current_index, 0, max_idx))
        actual_min = int(np.min(self._scan_file_indices))
        actual_max = int(np.max(self._scan_file_indices))
        self.indexSlider.blockSignals(True)
        self.indexSpin.blockSignals(True)
        self.indexSlider.setRange(actual_min, actual_max)
        self.indexSpin.setRange(actual_min, actual_max)
        cur_actual = int(self._scan_file_indices[cur])
        self.indexSlider.setValue(cur_actual)
        self.indexSpin.setValue(cur_actual)
        self.indexSlider.blockSignals(False)
        self.indexSpin.blockSignals(False)
        self.current_index = cur

    def _prepare_pointcloud_support(self):
        # Extrinsics
        try:
            self.extrinsics = load_extrinsics(EXTRINSICS_FIXED_PATH, verbose=False)
        except Exception:
            self.extrinsics = np.eye(4, dtype=np.float64)

        # Interpolators
        try:
            self.interps = build_interpolators(self.df_gps, verbose=False)
        except Exception:
            self.interps = None

        # Origin shift (first GNSS to (0,0,0))
        p0 = self.df_gps[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)[0]
        self.origin_shift = -p0

        # full polyline (shifted)
        P = self.df_gps[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
        P = P + self.origin_shift[None, :]
        self.polyline_full = P

        # resampled for markers
        self.resampled = resample_poses(self.df_gps, target_rate_hz=10.0, verbose=False)

    # ---- Point cloud updates ----
    def _update_pointcloud(self, force: bool = False):
        if self.centerStack.currentIndex() != 1 and not force:
            return
        if self.df_gps is None or self.scans is None or self.interps is None:
            return
        if gl is None:
            return

        self._pstart()
        self._plog("start _update_pointcloud")

        # Build chunk scans
        t_sel0 = time.perf_counter()
        start_idx = int(np.clip(self.current_index, 0, len(self.scans) - 1))
        k = max(1, int(self.index_interval))
        max_frames = max(1, int(self.max_frames))
        sel = list(range(start_idx, min(len(self.scans), start_idx + max_frames * k), k))
        chunk_scans = [self.scans[i] for i in sel]
        self._plog(f"select scans: {(time.perf_counter()-t_sel0)*1000:.1f} ms | {len(chunk_scans)} scans")

        # time bounds
        t_min = float(self.df_gps["t"].iloc[0])
        t_max = float(self.df_gps["t"].iloc[-1])

        # GPS-based offset
        gps_offset_s = float(self.offset_ms) * 1e-3
        lidar_time_offset_s = -gps_offset_s

        # representative time & center
        t_hi0 = time.perf_counter()
        t_choice = None
        for sc in chunk_scans:
            t_adj = float(sc.t) + lidar_time_offset_s
            if t_min <= t_adj <= t_max:
                t_choice = t_adj
                break
        stride = max(1, int(self.marker_stride))
        times_sub = self.resampled.times[::stride]
        positions_sub = (self.resampled.positions + self.origin_shift[None, :])[::stride]
        center_world = np.zeros(3, dtype=np.float64)
        hi = None
        if t_choice is not None and times_sub.size > 0:
            hi = int(np.argmin(np.abs(times_sub - t_choice)))
            center_world = positions_sub[hi].copy()
        self._plog(f"compute highlight/center: {(time.perf_counter()-t_hi0)*1000:.1f} ms")

        # dynamic adjust and filters
        T_world_adjust_dyn = np.array([
            [1, 0, 0, float(self.origin_shift[0] - center_world[0])],
            [0, 1, 0, float(self.origin_shift[1] - center_world[1])],
            [0, 0, 1, float(self.origin_shift[2] - center_world[2])],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        xR = float(self.x_range) if self.range_enabled else np.inf
        yR = float(self.y_range) if self.range_enabled else np.inf
        zR = float(self.z_range) if self.range_enabled else np.inf
        max_pts = None if int(self.max_points) <= 0 else int(self.max_points)

        # Accumulate
        t_acc0 = time.perf_counter()
        acc = accumulate_lidar_points(
            scans=chunk_scans,
            t_min=t_min, t_max=t_max,
            time_offset=lidar_time_offset_s,
            interps=self.interps,
            T_base_link_lidar=self.extrinsics,
            T_world_adjust=T_world_adjust_dyn,
            max_points=max_pts,
            x_range=xR, y_range=yR, z_range=zR,
            verbose=False,
        )
        self._plog(
            f"accumulate: {(time.perf_counter()-t_acc0)*1000:.1f} ms | pts={acc.points_xyz.shape[0]} scans_used={acc.num_scans_used}"
        )

        # GL updates
        t_gl0 = time.perf_counter()
        self.pcView.clear_items()
        if acc.points_xyz is None or acc.points_xyz.size == 0:
            self._plog("no points to render")
            return
        self.pcView.set_pointcloud(acc.points_xyz)
        self._plog(f"set_pointcloud total: {(time.perf_counter()-t_gl0)*1000:.1f} ms")

        t_mk0 = time.perf_counter()
        poly_shifted = self.polyline_full - center_world[None, :]
        start_shifted = poly_shifted[0] if poly_shifted.shape[0] > 0 else None
        marker_points = positions_sub - center_world[None, :]
        from scipy.spatial.transform import Rotation
        dirs = Rotation.from_quat(self.resampled.quaternions[::stride]).apply(np.array([1.0, 0.0, 0.0]))
        # Compute BEV-matched colors for markers using selected metric
        marker_colors = None
        marker_ci_vals = None
        try:
            if self._ci_all_full is not None and self._bev_t_full is not None:
                mi = int(self.selected_cov_idx)
                ci_full = np.asarray(self._ci_all_full[mi], dtype=np.float64)
                t_full = np.asarray(self._bev_t_full, dtype=np.float64)
                if t_full.size == ci_full.size and t_full.size > 1:
                    ci_interp = np.interp(times_sub, t_full, ci_full, left=ci_full[0], right=ci_full[-1])
                    marker_ci_vals = ci_interp.astype(np.float64)
                    ci_interp = np.clip(ci_interp, 1e-12, None)
                    norm = self._bev_norms[mi]
                    vals01 = norm(ci_interp)
                    cmap = plt.get_cmap(self.cmap)
                    rgba = cmap(vals01)
                    marker_colors = rgba.astype(np.float32)
        except Exception as e:
            self._plog(f"marker color map failed: {e}")

        # Compute LiDAR indices for each marker time (GPS-based offset)
        marker_lidar_idx = None
        try:
            if self._lidar_times is not None and self._lidar_times.size > 0 and times_sub.size > 0:
                adjusted = self._lidar_times + lidar_time_offset_s
                # nearest index via searchsorted
                j = np.searchsorted(adjusted, times_sub)
                j = np.clip(j, 1, adjusted.size - 1)
                prev = j - 1
                next_ = j
                choose_next = np.abs(adjusted[next_] - times_sub) < np.abs(adjusted[prev] - times_sub)
                marker_lidar_idx = np.where(choose_next, next_, prev).astype(int)
        except Exception as e:
            self._plog(f"lidar index map failed: {e}")
        self.pcView.set_polyline_and_markers(
            poly_shifted, start_shifted, marker_points, dirs, hi,
            marker_colors, marker_ci_vals, times_sub, marker_lidar_idx
        )
        self._plog(f"markers/polyline: {(time.perf_counter()-t_mk0)*1000:.1f} ms")

# ---- App entry ----

def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1280, 900)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()