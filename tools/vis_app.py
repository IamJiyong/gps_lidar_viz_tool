#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
import matplotlib.pyplot as plt

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


EXTRINSICS_FIXED_PATH = "extrinsics.yaml"


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
        self.setMouseTracking(True)  # <- 추가

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
        # 1) 남아있는 컬러바/보조 축 제거
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        # Matplotlib이 컬러바용으로 추가했던 보조 axes들 제거
        for ax_extra in list(self.fig.axes):
            if ax_extra is not self.ax:
                try:
                    self.fig.delaxes(ax_extra)
                except Exception:
                    pass

        # 2) 축 클리어 후 다크 테마 재적용
        self.ax.clear()
        self.fig.patch.set_facecolor(self._fig_bg)
        self.ax.set_facecolor(self._ax_bg)
        for spine in self.ax.spines.values():
            spine.set_color("w")

        # 3) 본 플롯
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

        # 4) 컬러바 새로 추가 (기존 축만 유지되도록)
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
        self.fig.canvas.setMouseTracking(True)  # <- 추가
        self.setAttribute(QtCore.Qt.WA_Hover, True)  # <- 추가
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
        # Matplotlib toolbar 우측 좌표 문자열 커스터마이즈
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
                fontsize=12, color="w"  # <- 12pt로 확대
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

        # focus for key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def eventFilter(self, obj, ev):
        if obj is self.view and ev.type() == QtCore.QEvent.KeyPress:
            key = ev.key()
            if key == Qt.Key_A:
                self.indexChanged.emit(-self._idx_stride); ev.accept(); return True
            elif key == Qt.Key_D:
                self.indexChanged.emit(+self._idx_stride); ev.accept(); return True
            elif key == Qt.Key_Q:
                self.offsetChangedMs.emit(-self._offset_stride_ms); ev.accept(); return True
            elif key == Qt.Key_E:
                self.offsetChangedMs.emit(+self._offset_stride_ms); ev.accept(); return True
        return super().eventFilter(obj, ev)

    def set_strides(self, idx_stride: int, offset_stride_ms: int):
        self._idx_stride = max(1, int(idx_stride))
        self._offset_stride_ms = int(offset_stride_ms)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_A:
            self.indexChanged.emit(-self._idx_stride); ev.accept(); return
        elif key == Qt.Key_D:
            self.indexChanged.emit(+self._idx_stride); ev.accept(); return
        elif key == Qt.Key_Q:
            self.offsetChangedMs.emit(-self._offset_stride_ms); ev.accept(); return
        elif key == Qt.Key_E:
            self.offsetChangedMs.emit(+self._offset_stride_ms); ev.accept(); return
        super().keyPressEvent(ev)

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
                                 highlight_idx: Optional[int]):
        if gl is None:
            return

        # Polyline (밝은 노란색, 시인성 향상)
        if polyline_xyz is not None and polyline_xyz.shape[0] >= 2:
            plt3d = gl.GLLinePlotItem(pos=polyline_xyz, color=(0.85, 0.85, 0.20, 1.0),
                                      width=2.0, antialias=True)
            self.polyline_item = plt3d
            self.view.addItem(plt3d)

        # Start sphere (빨강)
        if start_point is not None:
            sp = self._make_sphere(start_point, radius=0.35, color=(1, 0, 0, 1))
            self.start_sphere = sp
            self.view.addItem(sp)

        # Marker arrows (cones) — 일반 마커: 밝은 청록
        self.marker_meshes.clear()
        if marker_points is not None and marker_dirs is not None and marker_points.shape[0] == marker_dirs.shape[0]:
            for i in range(marker_points.shape[0]):
                p = marker_points[i]
                d = marker_dirs[i]
                cone = self._make_cone(p, d, length=0.8, radius=0.06, color=(0.20, 0.85, 0.95, 1.0))
                cone.setGLOptions('opaque')  # 깊이/불투명(기본)
                self.marker_meshes.append(cone)
                self.view.addItem(cone)

        # Highlight arrow — 선명한 주황/빨강, 가장 마지막에 추가 + additive로 최상단 가시성
        if highlight_idx is not None and 0 <= highlight_idx < marker_points.shape[0]:
            p = marker_points[highlight_idx]
            d = marker_dirs[highlight_idx]
            self.highlight_mesh = self._make_cone(p, d, length=1.2, radius=0.08, color=(1.0, 0.35, 0.05, 1.0))
            self.highlight_mesh.setGLOptions('additive')  # 블렌딩 우선으로 다른 메쉬 위에 보이게
            self.view.addItem(self.highlight_mesh)

        if not self._has_topdown_set:
            self.set_topdown_camera()

    def set_pointcloud(self, xyz: np.ndarray):
        if gl is None:
            return
        if xyz is None or xyz.size == 0:
            return
        z = xyz[:, 2]
        zmin = float(np.min(z)) if z.size else 0.0
        zmax = float(np.max(z)) if z.size else 1.0
        if zmax > zmin:
            zn = (z - zmin) / (zmax - zmin)
        else:
            zn = np.zeros_like(z)
        colors = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        for i, h in enumerate((2.0 / 3.0) * (1.0 - zn)):
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
            colors[i] = [r, g, b, 1.0]

        # 아주 작게: 픽셀 모드 + 작은 사이즈
        sp = gl.GLScatterPlotItem(pos=xyz.astype(np.float32), color=colors, size=1.0, pxMode=True)
        if self.cloud_item is not None:
            self.view.removeItem(self.cloud_item)
        self.cloud_item = sp
        self.view.addItem(sp)


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

        self.reloadBtn = QtWidgets.QPushButton("Reload Point Cloud")
        self.reloadBtn.clicked.connect(self.reloadRequested.emit)

        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(btnBox); lay.addWidget(self.reloadBtn)

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
        self.extrinsics = None
        self.interps = None
        self.resampled = None
        self.polyline_full = None
        self.origin_shift = np.zeros(3, dtype=np.float64)

        # BEV state
        self.cov_cols = COV_COLS_DEFAULT
        self.cmap = "viridis"
        self.point_size = 6.0
        self.norm_mode = "per"
        self._bev_stride = 1

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
        # Reload 버튼 동작: 현재 다이얼로그 값 적용 후 즉시 리로드 (다이얼로그는 유지)
        dlg.reloadRequested.connect(lambda: self._apply_options_and_reload(dlg))

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            v = dlg.values()
            self._apply_options_from_values(v)

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
        self.current_index = int(val)
        self.indexSpin.blockSignals(True)
        self.indexSpin.setValue(int(val))
        self.indexSpin.blockSignals(False)
        self._update_pointcloud()

    def _on_index_spin(self, val: int):
        # This handler is now only for the slider's valueChanged signal
        pass

    def _on_index_spin_commit(self):
        val = int(self.indexSpin.value())
        if self.scans is None:
            return
        val = int(np.clip(val, 0, len(self.scans) - 1))
        if val != self.current_index:
            self.current_index = val
            self.indexSlider.blockSignals(True)
            self.indexSpin.blockSignals(True)
            self.indexSlider.setValue(val)
            self.indexSpin.setValue(val)
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
            self.indexSlider.blockSignals(True)
            self.indexSpin.blockSignals(True)
            self.indexSlider.setValue(new_idx)
            self.indexSpin.setValue(new_idx)
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
        total = len(self.scans)
        self.indexSlider.blockSignals(True)
        self.indexSpin.blockSignals(True)
        self.indexSlider.setRange(0, max(0, total - 1))
        self.indexSpin.setRange(0, max(0, total - 1))
        self.indexSlider.setValue(0)
        self.indexSpin.setValue(0)
        self.indexSlider.blockSignals(False)
        self.indexSpin.blockSignals(False)
        self.current_index = 0

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

        # Build chunk scans
        start_idx = int(np.clip(self.current_index, 0, len(self.scans) - 1))
        k = max(1, int(self.index_interval))
        max_frames = max(1, int(self.max_frames))
        sel = list(range(start_idx, min(len(self.scans), start_idx + max_frames * k), k))
        chunk_scans = [self.scans[i] for i in sel]

        # time bounds
        t_min = float(self.df_gps["t"].iloc[0])
        t_max = float(self.df_gps["t"].iloc[-1])

        # GPS-based offset
        gps_offset_s = float(self.offset_ms) * 1e-3
        lidar_time_offset_s = -gps_offset_s

        # 대표 시간(t_choice) 및 현재 중심(center_world, origin_shift 적용 좌표계)
        t_choice = None
        for sc in chunk_scans:
            t_adj = float(sc.t) + lidar_time_offset_s
            if t_min <= t_adj <= t_max:
                t_choice = t_adj
                break

        # resampled/markers 준비
        stride = max(1, int(self.marker_stride))
        times_sub = self.resampled.times[::stride]
        positions_sub = (self.resampled.positions + self.origin_shift[None, :])[::stride]

        center_world = np.zeros(3, dtype=np.float64)
        hi = None
        if t_choice is not None and times_sub.size > 0:
            hi = int(np.argmin(np.abs(times_sub - t_choice)))
            center_world = positions_sub[hi].copy()

        # 범위 중심을 현재 시점으로 이동: T_world_adjust_dyn = origin_shift - center_world
        T_world_adjust_dyn = np.array([
            [1, 0, 0, float(self.origin_shift[0] - center_world[0])],
            [0, 1, 0, float(self.origin_shift[1] - center_world[1])],
            [0, 0, 1, float(self.origin_shift[2] - center_world[2])],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Filters (range_enabled일 때만 활성)
        xR = float(self.x_range) if self.range_enabled else np.inf
        yR = float(self.y_range) if self.range_enabled else np.inf
        zR = float(self.z_range) if self.range_enabled else np.inf
        max_pts = None if int(self.max_points) <= 0 else int(self.max_points)

        # Accumulate (현재 중심 기준 필터)
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

        # 시각화 좌표: polyline/markers도 동일하게 중심 이동(-center_world)
        self.pcView.clear_items()
        if acc.points_xyz is None or acc.points_xyz.size == 0:
            return
        self.pcView.set_pointcloud(acc.points_xyz)

        poly_shifted = self.polyline_full - center_world[None, :]
        start_shifted = poly_shifted[0] if poly_shifted.shape[0] > 0 else None
        marker_points = positions_sub - center_world[None, :]
        from scipy.spatial.transform import Rotation
        dirs = Rotation.from_quat(self.resampled.quaternions[::stride]).apply(np.array([1.0, 0.0, 0.0]))
        self.pcView.set_polyline_and_markers(poly_shifted, start_shifted, marker_points, dirs, hi)

# ---- App entry ----

def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1280, 900)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()