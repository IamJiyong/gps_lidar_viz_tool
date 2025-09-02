# tools/pointcloud_view.py
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import colors as mcolors

try:
    from OpenGL.GL import glGetDoublev, glGetIntegerv, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
except Exception:
    glGetDoublev = None; glGetIntegerv = None; GL_MODELVIEW_MATRIX = None; GL_PROJECTION_MATRIX = None; GL_VIEWPORT = None


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
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)
        self.setFocusProxy(self.view)

        # wrap in a frame to allow reliable border styling
        self.frame = QtWidgets.QFrame()
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        fl = QtWidgets.QVBoxLayout(self.frame); fl.setContentsMargins(0,0,0,0); fl.addWidget(self.view, 1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.frame, 1)

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
        self._pan_speed_ratio = 1.0
        self._zoom_rate_in_per_sec = 0.2
        self._zoom_rate_out_per_sec = 1.0 / self._zoom_rate_in_per_sec
        self._dist_min = 0.05
        self._dist_max = 1e6

        self._pan_dx = 0
        self._pan_dy = 0
        self._zoom_dir = 0
        self._motion_timer = QtCore.QTimer(self)
        self._motion_timer.setInterval(16)
        self._motion_timer.timeout.connect(self._on_motion_tick)
        self._last_motion_ts = time.perf_counter()

        self.profile_cb: Optional[Callable[[str], None]] = None

        self._marker_points = None
        self._marker_ci_vals = None
        self._marker_times = None
        self._marker_lidar_idx = None
        self._hover_last_idx = None
        self._marker_dirs = None
        self._marker_lengths = None

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
                    try:
                        pos = ev.pos()
                    except Exception:
                        return False
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
                    try:
                        pos = ev.pos()
                    except Exception:
                        return False
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

    def fit_to_extent(self, xyz: np.ndarray, set_distance: bool = False) -> None:
        if xyz is None or xyz.size == 0:
            return
        ext = np.ptp(xyz, axis=0).astype(float)
        diag = float(np.linalg.norm(ext) + 1e-6)
        target_dist = max(5.0, 0.9 * diag)

        # Scale controls with scene size
        self._pan_speed_ratio = 0.25
        self._dist_min = max(0.01, 1e-3 * diag)
        self._dist_max = max(target_dist * 50.0, 50.0)
        self._zoom_rate_in_per_sec = 0.4
        self._zoom_rate_out_per_sec = 1.0 / self._zoom_rate_in_per_sec

        # Preserve current view distance by default; clamp to new limits
        if set_distance:
            self.view.opts['distance'] = target_dist
        else:
            prev = float(self.view.opts.get('distance', target_dist))
            self.view.opts['distance'] = float(np.clip(prev, self._dist_min, self._dist_max))

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

    def set_filter_border(self, enabled: bool):
        try:
            if enabled:
                self.frame.setStyleSheet("QFrame{border:3px solid #ff4d4d;}")
            else:
                self.frame.setStyleSheet("")
        except Exception:
            pass

    def set_pointcloud(self, xyz: np.ndarray, *, color_mode: str = "default", intensities: Optional[np.ndarray] = None, per_scan_offsets: Optional[List[int]] = None):
        if gl is None:
            return
        if xyz is None or xyz.size == 0:
            return
        t0 = time.perf_counter()
        colors = None
        mode = (color_mode or "default").lower()
        if mode == "intensity" and intensities is not None and intensities.size == xyz.shape[0]:
            vals = intensities.astype(np.float64)
            vals = vals[np.isfinite(vals)] if vals.size else vals
            if vals.size:
                p1 = float(np.percentile(vals, 1.0))
                p99 = float(np.percentile(vals, 99.0))
                span = max(1e-12, p99 - p1)
                vi = np.clip(intensities, p1, p99)
                vn = (vi - p1) / span
            else:
                vn = np.zeros_like(intensities, dtype=np.float64)
            rgb = mcolors.hsv_to_rgb(np.stack([(2.0/3.0)*(1.0 - vn), np.ones_like(vn), np.ones_like(vn)], axis=1)).astype(np.float32)
            colors = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        elif mode == "per-scan" and per_scan_offsets is not None and len(per_scan_offsets) > 0:
            ids = np.zeros(xyz.shape[0], dtype=np.int32)
            offsets = list(per_scan_offsets) + [xyz.shape[0]]
            for k in range(len(per_scan_offsets)):
                ids[offsets[k]:offsets[k+1]] = k
            if len(per_scan_offsets) > 1:
                vn = ids.astype(np.float64) / float(len(per_scan_offsets) - 1)
            else:
                vn = np.zeros_like(ids, dtype=np.float64)
            rgb = mcolors.hsv_to_rgb(np.stack([(2.0/3.0)*(1.0 - vn), np.ones_like(vn), np.ones_like(vn)], axis=1)).astype(np.float32)
            colors = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            z = xyz[:, 2]
            zmin = float(np.min(z)) if z.size else 0.0
            zmax = float(np.max(z)) if z.size else 1.0
            if zmax > zmin:
                zn = (z - zmin) / (zmax - zmin)
            else:
                zn = np.zeros_like(z)
            hues = (2.0 / 3.0) * (1.0 - zn)
            hsv = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1)
            rgb = mcolors.hsv_to_rgb(hsv).astype(np.float32)
            colors = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        if self.profile_cb:
            self.profile_cb(f"color mapping: {(time.perf_counter()-t0)*1000:.1f} ms for {xyz.shape[0]} pts (mode={mode})")

        t1 = time.perf_counter()
        sp = gl.GLScatterPlotItem(pos=xyz.astype(np.float32), color=colors, size=1.0, pxMode=True)
        if self.cloud_item is not None:
            self.view.removeItem(self.cloud_item)
        self.cloud_item = sp
        self.view.addItem(sp)
        if self.profile_cb:
            self.profile_cb(f"GL scatter upload: {(time.perf_counter()-t1)*1000:.1f} ms")