#!/usr/bin/env python3
from __future__ import annotations

import os
import time
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

# Local UI components
from bev_canvas import BEVClickableCanvas, BEVMainCanvas
from pointcloud_view import PointCloudView
from options_dialog import OptionsDialog


EXTRINSICS_FIXED_PATH = "extrinsics.yaml"


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
        self._prof_enabled = False
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

    def _build_ui(self):
        toolbar = QtWidgets.QToolBar("Main")
        self.addToolBar(toolbar)

        actOpenGps = QtWidgets.QAction("Open GPS CSV", self)
        actOpenLidar = QtWidgets.QAction("Open LiDAR Dir", self)
        actOpenFolder = QtWidgets.QAction("Open Folder", self)

        actOptions = QtWidgets.QAction("Options", self)
        actViewCloud = QtWidgets.QAction("View Point Cloud", self)
        actViewCloud.setCheckable(True)
        actViewCloud.setChecked(False)

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

        actOpenGps.triggered.connect(self._on_open_gps)
        actOpenLidar.triggered.connect(self._on_open_lidar)
        actOpenFolder.triggered.connect(self._on_open_folder)
        actOptions.triggered.connect(self._on_options)

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
        self.pcView.indexChanged.connect(self._on_index_step)
        self.pcView.offsetChangedMs.connect(self._on_offset_step_ms)
        self.pcView.profile_cb = self._plog
        self.centerStack.addWidget(self.pcView)

        # Right panel
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

        ctrlGroup = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrlGroup)

        # Metric selection
        self.metricCombo = QtWidgets.QComboBox()
        self.metricCombo.setEnabled(False)
        self.metricCombo.currentIndexChanged.connect(self._on_metric_changed)
        form.addRow("Metric", self.metricCombo)

        # Time offset
        self.offsetSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.offsetSlider.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSlider.setSingleStep(int(self.offset_slider_step_ms))
        self.offsetSlider.setValue(int(self.offset_ms))
        self.offsetSpin = QtWidgets.QSpinBox()
        self.offsetSpin.setRange(int(self.offset_min_ms), int(self.offset_max_ms))
        self.offsetSpin.setSingleStep(1)
        self.offsetSpin.setSuffix(" ms")
        self.offsetSpin.setValue(int(self.offset_ms))
        self.offsetSpin.setKeyboardTracking(False)
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

        # LiDAR index
        self.indexSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.indexSlider.setRange(0, 0)
        self.indexSlider.setValue(0)
        self.indexSpin = QtWidgets.QSpinBox()
        self.indexSpin.setRange(0, 0)
        self.indexSpin.setValue(0)
        self.indexSpin.setKeyboardTracking(False)
        self.indexSpin.lineEdit().returnPressed.connect(self._on_index_spin_commit)

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

            QGroupBox { border: 1px solid #3b3f45; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; color: #e6e6e6; }
            QLabel { color: #e6e6e6; }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #1f2023;
                color: #e6e6e6;
                border: 1px solid #3b3f45;
            }
            QComboBox {
                color: #000000;
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

            QSlider::groove:horizontal { background: #3b3f45; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e6e6e6; width: 12px; margin: -4px 0; border-radius: 6px; }

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

        # Global shortcuts (active only on point cloud view)
        from PyQt5 import QtGui
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

        if self.df_gps is not None:
            self._prepare_pointcloud_support()

    def _auto_find_in_folder(self, root_dir: str) -> Tuple[Optional[str], Optional[str]]:
        candidates_csv = [
            os.path.join(root_dir, "GPS", "Odom_data.csv"),
            os.path.join(root_dir, "GPS_data.csv"),
            os.path.join(root_dir, "Odom_data.csv"),
        ]
        csv_path = next((p for p in candidates_csv if os.path.isfile(p)), None)

        if csv_path is None:
            for dirpath, _dirnames, filenames in os.walk(root_dir):
                for name in filenames:
                    lower = name.lower()
                    if lower in ("odom_data.csv", "gps_data.csv"):
                        csv_path = os.path.join(dirpath, name)
                        break
                if csv_path:
                    break

        lidar_dir = None
        if os.path.isdir(os.path.join(root_dir, "lidar_xyzi")):
            lidar_dir = os.path.join(root_dir, "lidar_xyzi")
        else:
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
            self.pcView.view.setFocus()
            self.pcView.activateWindow()
            self.pcView.raise_()
            self.pcView.set_strides(self.index_stride, int(self.offset_stride_ms))
            self._refresh_index_bounds()
            self._update_pointcloud(force=True)

    # ---- Control handlers ----
    def _on_offset_slider(self, val: int):
        self.offset_ms = float(val)
        self.offsetSpin.blockSignals(True)
        self.offsetSpin.setValue(int(val))
        self.offsetSpin.blockSignals(False)
        self._update_pointcloud()

    def _on_offset_spin(self, val: int):
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
        x_m, y_m, df2 = xy_from_df(self.df_gps, origin_lat=None, origin_lon=None)
        self._bev_x, self._bev_y = select_stride(x_m, y_m, self._bev_stride)
        self._bev_df = df2.reset_index(drop=True)
        self._bev_t = self._bev_df["t"].to_numpy(dtype=np.float64) if "t" in self._bev_df.columns else None

        ci_all = compute_ci_arrays(self._bev_df, self.cov_cols)
        self._bev_ci_vals = [select_stride(r.ci_values, r.ci_values, self._bev_stride)[0] for r in ci_all]
        self._bev_units = [r.unit for r in ci_all]
        self._bev_norms = build_norms(ci_all, mode=self.norm_mode)
        self._ci_all_full = [np.asarray(r.ci_values, dtype=np.float64) for r in ci_all]
        self._bev_t_full = self._bev_df["t"].to_numpy(dtype=np.float64) if "t" in self._bev_df.columns else None

        self.metricCombo.blockSignals(True)
        self.metricCombo.clear()
        self.metricCombo.addItems(self.cov_cols)
        self.metricCombo.setEnabled(True)
        self.metricCombo.setCurrentIndex(int(self.selected_cov_idx))
        self.metricCombo.blockSignals(False)

        self._build_thumbnails()
        self._show_bev_idx(0)

    def _build_thumbnails(self):
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

        start_xy = (self._bev_x[0], self._bev_y[0]) if self._bev_x.size > 0 else None
        for i, canvas in enumerate(self.thumb_canvases):
            canvas.draw_scatter(
                self._bev_x, self._bev_y, self._bev_ci_vals[i],
                title=self.cov_cols[i], unit=self._bev_units[i], cmap=self.cmap,
                norm=self._bev_norms[i], point_size=self.point_size,
                start_xy=start_xy, show_axes_labels=False
            )

        self.thumb_layout.addStretch(1)

    @QtCore.pyqtSlot(int)
    def _show_bev_idx(self, idx: int):
        if idx < 0 or idx >= len(self.cov_cols):
            return
        self.main_canvas.reset_hover()
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
        t_vals = self._bev_t if self._bev_t is not None and len(self._bev_t) == len(self._bev_x) else np.arange(len(self._bev_x), dtype=float)
        self.main_canvas.set_hover_data(
            self._bev_x, self._bev_y, t_vals, self._bev_ci_vals[idx], self._bev_units[idx], self.cov_cols[idx],
            lidar_index_resolver=self._lidar_index_for_gps_time
        )
        if self.centerStack.currentIndex() == 1:
            self._update_pointcloud(force=True)

    def _on_metric_changed(self, idx: int):
        if idx < 0 or idx >= len(self.cov_cols):
            return
        self._show_bev_idx(int(idx))

    def _lidar_index_for_gps_time(self, t_gps: float) -> Optional[int]:
        if self._lidar_times is None or self._lidar_times.size == 0:
            return None
        gps_offset_s = float(self.offset_ms) * 1e-3
        lidar_time_offset_s = -gps_offset_s
        adjusted = self._lidar_times + lidar_time_offset_s
        i = int(np.argmin(np.abs(adjusted - float(t_gps))))
        return i

    def _refresh_index_bounds(self):
        if self.scans is None:
            return
        self._scan_file_indices = np.array([s.index for s in self.scans], dtype=int)
        total = len(self.scans)
        max_idx = max(0, total - 1)
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
        try:
            self.extrinsics = load_extrinsics(EXTRINSICS_FIXED_PATH, verbose=False)
        except Exception:
            self.extrinsics = np.eye(4, dtype=np.float64)

        try:
            self.interps = build_interpolators(self.df_gps, verbose=False)
        except Exception:
            self.interps = None

        p0 = self.df_gps[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)[0]
        self.origin_shift = -p0

        P = self.df_gps[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
        P = P + self.origin_shift[None, :]
        self.polyline_full = P

        self.resampled = resample_poses(self.df_gps, target_rate_hz=10.0, verbose=False)

    # ---- Point cloud updates ----
    def _update_pointcloud(self, force: bool = False):
        if self.centerStack.currentIndex() != 1 and not force:
            return
        if self.df_gps is None or self.scans is None or self.interps is None:
            return
        try:
            import pyqtgraph.opengl as gl  # runtime guard
        except Exception:
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

        # Markers/polyline
        t_mk0 = time.perf_counter()
        poly_shifted = self.polyline_full - center_world[None, :]
        start_shifted = poly_shifted[0] if poly_shifted.shape[0] > 0 else None
        marker_points = positions_sub - center_world[None, :]
        from scipy.spatial.transform import Rotation
        dirs = Rotation.from_quat(self.resampled.quaternions[::stride]).apply(np.array([1.0, 0.0, 0.0]))

        # BEV-matched colors
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

        # LiDAR indices for markers
        marker_lidar_idx = None
        try:
            if self._lidar_times is not None and self._lidar_times.size > 0 and times_sub.size > 0:
                adjusted = self._lidar_times + lidar_time_offset_s
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


def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1280, 900)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()