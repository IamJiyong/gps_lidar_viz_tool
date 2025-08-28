#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple, Callable, Dict

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui
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
from image_panel import ImagePanel


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

        # Camera image state
        self.cam_files: Optional[List[List[str]]] = None   # 6 lists (cam1..6)
        self.cam_times: Optional[List[np.ndarray]] = None  # 6 arrays (float seconds)
        self.cam_loaded_root: Optional[str] = None
        # default offsets (ms): cam1=100, others=500
        self.cam_offsets_ms = np.array([100.0, 500.0, 500.0, 500.0, 500.0, 500.0], dtype=np.float64)
        self._selected_cam_1based = 1
        self._pix_cache: dict = {}  # path -> QPixmap

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
        actOpenImages = QtWidgets.QAction("Open Images Folder", self)

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
        openMenu.addAction(actOpenImages)
        openBtn.setMenu(openMenu)

        toolbar.addWidget(openBtn)
        toolbar.addSeparator()
        toolbar.addAction(actOptions)

        actOpenGps.triggered.connect(self._on_open_gps)
        actOpenLidar.triggered.connect(self._on_open_lidar)
        actOpenFolder.triggered.connect(self._on_open_folder)
        actOpenImages.triggered.connect(self._on_open_images_folder)
        actOptions.triggered.connect(self._on_options)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        self.setCentralWidget(splitter)

        # Left image panel (always visible)
        self.imagePanel = ImagePanel()
        self.imagePanel.set_labels(["Front-Left", "Front", "Front-Right", "Rear-Left", "Rear", "Rear-Right"])
        self.imagePanel.thumbnailClicked.connect(self._on_thumb_clicked)
        self.imagePanel.vsplit.setHandleWidth(8)

        # Center: Point cloud view only
        self.centerStack = QtWidgets.QStackedWidget()
        self.pcView = PointCloudView()
        self.pcView.indexChanged.connect(self._on_index_step)
        self.pcView.offsetChangedMs.connect(self._on_offset_step_ms)
        self.pcView.profile_cb = self._plog
        self.centerStack.addWidget(self.pcView)

        # Right panel with vertical splitter: top main BEV plot, bottom thumbnails + controls
        rightPanel = QtWidgets.QWidget()
        rightV = QtWidgets.QVBoxLayout(rightPanel); rightV.setContentsMargins(6,6,6,6); rightV.setSpacing(8)

        rightSplit = QtWidgets.QSplitter(Qt.Vertical)
        rightSplit.setHandleWidth(8)

        # Top: BEV main plot
        bevTop = QtWidgets.QWidget()
        bevTopV = QtWidgets.QVBoxLayout(bevTop); bevTopV.setContentsMargins(0,0,0,0); bevTopV.setSpacing(4)
        self.main_canvas = BEVMainCanvas(bevTop)
        self.main_toolbar = NavToolbar(self.main_canvas, bevTop)
        bevTopV.addWidget(self.main_toolbar)
        bevTopV.addWidget(self.main_canvas, 1)

        # Bottom: thumbnails + controls
        rightBottom = QtWidgets.QWidget()
        rbV = QtWidgets.QVBoxLayout(rightBottom); rbV.setContentsMargins(0,0,0,0); rbV.setSpacing(8)

        self.thumb_container = QtWidgets.QWidget()
        # Use a fixed 2x3 grid layout for thumbnails
        self.thumb_grid = QtWidgets.QGridLayout(self.thumb_container)
        self.thumb_grid.setContentsMargins(0, 0, 0, 0)
        self.thumb_grid.setSpacing(8)
        self.thumb_grid.setColumnStretch(0, 1)
        self.thumb_grid.setColumnStretch(1, 1)
        self.thumb_grid.setColumnStretch(2, 1)
        self.thumb_grid.setRowStretch(0, 1)
        self.thumb_grid.setRowStretch(1, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.thumb_container)
        rbV.addWidget(scroll, 1)

        ctrlGroup = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrlGroup)

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

        # View Point Cloud toggle button (kept; focuses the pc view)
        self.viewCloudToggle = QtWidgets.QPushButton("View Point Cloud")
        self.viewCloudToggle.setCheckable(True)
        self.viewCloudToggle.setChecked(False)
        self.viewCloudToggle.toggled.connect(self._on_toggle_cloud)
        rbV.addWidget(self.viewCloudToggle, 0)

        rbV.addWidget(ctrlGroup, 0)

        rightSplit.addWidget(bevTop)
        rightSplit.addWidget(rightBottom)
        rightV.addWidget(rightSplit, 1)

        splitter.addWidget(self.imagePanel)
        splitter.addWidget(self.centerStack)
        splitter.addWidget(rightPanel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([420, 900, 480])

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

            /* Splitter handle visibility */
            QSplitter::handle {
                background-color: #5a5e66;
            }
            QSplitter::handle:horizontal {
                width: 10px;
            }
            QSplitter::handle:vertical {
                height: 10px;
            }
            QSplitter::handle:hover {
                background-color: #7a7f88;
            }
            QSplitter::handle:pressed {
                background-color: #9aa0aa;
            }
        """)

        # Global shortcuts (active only on point cloud view)
        from PyQt5 import QtGui
        self.shortA = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self); self.shortA.setContext(Qt.ApplicationShortcut)
        self.shortD = QtWidgets.QShortcut(QtGui.QKeySequence("D"), self); self.shortD.setContext(Qt.ApplicationShortcut)
        self.shortQ = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self); self.shortQ.setContext(Qt.ApplicationShortcut)
        self.shortE = QtWidgets.QShortcut(QtGui.QKeySequence("E"), self); self.shortE.setContext(Qt.ApplicationShortcut)
        self.shortA.activated.connect(lambda: self._on_index_step(-self.index_stride))
        self.shortD.activated.connect(lambda: self._on_index_step(+self.index_stride))
        self.shortQ.activated.connect(lambda: self._on_offset_step_ms(-self.offset_stride_ms))
        self.shortE.activated.connect(lambda: self._on_offset_step_ms(+self.offset_stride_ms))

        # Optional: print where focus sits when UI shows (helps detect if a QLineEdit is eating keys)
        QtCore.QTimer.singleShot(500, lambda: print(f"[focus] initial focus={QtWidgets.QApplication.focusWidget().metaObject().className() if QtWidgets.QApplication.focusWidget() else 'None'}"))

        # Also add explicit prints on shortcut activation to confirm signal paths:
        self.shortA.activated.connect(lambda: print("[shortcut] A activated"))
        self.shortD.activated.connect(lambda: print("[shortcut] D activated"))
        self.shortQ.activated.connect(lambda: print("[shortcut] Q activated"))
        self.shortE.activated.connect(lambda: print("[shortcut] E activated"))

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
    
    def _on_index_stride_changed(self, val: int):
        self.index_stride = max(1, int(val))
        self.pcView.set_strides(self.index_stride, int(self.offset_stride_ms))

    def _on_open_lidar(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open LiDAR Directory", "")
        if not d:
            return
        self.scans = parse_lidar_directory(d)
        self._lidar_times = np.array([s.t for s in self.scans], dtype=np.float64)
        self._refresh_index_bounds()

        if self.df_gps is not None:
            self._prepare_pointcloud_support()
        self._update_images()

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

    def _auto_find_cameras(self, root_dir: str) -> Optional[str]:
        # Accept either:
        #  - root_dir/camera_1..camera_6
        #  - root_dir/decoded_rgb/camera_1..camera_6
        candidates = os.path.join(root_dir, "decoded_rgb")
        names = [f"camera_{i}" for i in range(1, 7)]
        if all(os.path.isdir(os.path.join(candidates, n)) for n in names):
            return candidates
        return None

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

        # Cameras
        cam_root = self._auto_find_cameras(d)
        if cam_root:
            try:
                self._load_camera_set(cam_root)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Camera Error", str(e))
        else:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Could not find camera_1..camera_6 in the selected folder.")
 
    def _on_open_images_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Images Folder", "")
        if not d:
            return
        cam_root = self._auto_find_cameras(d)
        if not cam_root:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Could not find camera_1..camera_6 in the selected folder.")
            return
        try:
            self._load_camera_set(cam_root)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Camera Error", str(e))

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
            cam_offsets_ms=self.cam_offsets_ms.tolist() if isinstance(self.cam_offsets_ms, np.ndarray) else self.cam_offsets_ms,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            v = dlg.values()
            self._apply_options_from_values(v)
            self._update_pointcloud(force=True)
            self._update_pointcloud(force=True)
            self._update_images()

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
        if "cam_offsets_ms" in v:
            arr = np.array(v["cam_offsets_ms"], dtype=np.float64).reshape(-1)
            if arr.size == 6:
                self.cam_offsets_ms = arr
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
        self._update_images()

    def _on_toggle_cloud(self, checked: bool):
        # Center is point cloud view only; just focus when toggled on
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
        self._update_images()
        self._refresh_bev_markers()

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
            self._update_images()
            self._refresh_bev_markers()

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
            self._update_images()
            self._refresh_bev_markers()
            return
        arr = self._scan_file_indices
        i = int(np.argmin(np.abs(arr - int(val))))
        self.current_index = i
        snapped = int(arr[i])
        self.indexSpin.blockSignals(True)
        self.indexSpin.setValue(snapped)
        self.indexSpin.blockSignals(False)
        self._update_pointcloud()
        self._update_images()
        self._refresh_bev_markers()

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
                self._update_images()
                self._refresh_bev_markers()
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
            self._update_images()
            self._refresh_bev_markers()

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
            self._update_images()
            self._refresh_bev_markers()
            
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

        self._build_thumbnails()
        self._show_bev_idx(int(self.selected_cov_idx) if 0 <= int(self.selected_cov_idx) < len(self.cov_cols) else 0)

    def _build_thumbnails(self):
        # Clear previous items from the grid (if any)
        if hasattr(self, "thumb_grid") and isinstance(self.thumb_grid, QtWidgets.QGridLayout):
            while self.thumb_grid.count():
                item = self.thumb_grid.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
        else:
            # Fallback: if thumb_grid wasn't created yet, create it now
            self.thumb_grid = QtWidgets.QGridLayout(self.thumb_container)
            self.thumb_grid.setContentsMargins(0, 0, 0, 0)
            self.thumb_grid.setSpacing(8)
            self.thumb_grid.setColumnStretch(0, 1)
            self.thumb_grid.setColumnStretch(1, 1)
            self.thumb_grid.setColumnStretch(2, 1)
            self.thumb_grid.setRowStretch(0, 1)
            self.thumb_grid.setRowStretch(1, 1)

        # Build up to 6 thumbnails in a 2x3 grid
        self.thumb_canvases: List[BEVClickableCanvas] = []
        n_show = min(6, len(self.cov_cols))
        start_xy = (self._bev_x[0], self._bev_y[0]) if getattr(self, "_bev_x", None) is not None and self._bev_x.size > 0 else None
        cur_xy = self._current_bev_dot_xy()

        for i in range(n_show):
            canvas = BEVClickableCanvas(idx=i, parent=self.thumb_container, width=3.2, height=2.0, dpi=100)
            canvas.clicked.connect(self._show_bev_idx)
            self.thumb_canvases.append(canvas)
            r = 0 if i < 3 else 1
            c = i if i < 3 else (i - 3)
            self.thumb_grid.addWidget(canvas, r, c)

            # initial draw
            canvas.draw_scatter(
                self._bev_x, self._bev_y, self._bev_ci_vals[i],
                title=self.cov_cols[i], unit=self._bev_units[i], cmap=self.cmap,
                norm=self._bev_norms[i], point_size=self.point_size,
                start_xy=start_xy, show_axes_labels=False,
                current_xy=cur_xy
            )

    def _current_bev_dot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            if getattr(self, "_bev_x", None) is None or getattr(self, "_bev_y", None) is None:
                return None
            if self._bev_t is None or self._bev_x.size == 0:
                return None
            if self.scans is None or len(self.scans) == 0:
                return None
            idx = int(np.clip(self.current_index, 0, len(self.scans) - 1))
            t_lidar = float(self.scans[idx].t)
            gps_offset_s = float(self.offset_ms) * 1e-3
            t_gps = t_lidar - gps_offset_s
            t_vals = self._bev_t if self._bev_t is not None and len(self._bev_t) == len(self._bev_x) else np.arange(len(self._bev_x), dtype=float)
            j = int(np.argmin(np.abs(t_vals - t_gps)))
            return float(self._bev_x[j]), float(self._bev_y[j])
        except Exception:
            return None

    def _redraw_thumbnails_with_markers(self):
        if not hasattr(self, "thumb_canvases"):
            return
        if getattr(self, "_bev_x", None) is None or self._bev_x.size == 0:
            return
        start_xy = (self._bev_x[0], self._bev_y[0])
        cur_xy = self._current_bev_dot_xy()
        for i, canvas in enumerate(self.thumb_canvases):
            canvas.draw_scatter(
                self._bev_x, self._bev_y, self._bev_ci_vals[i],
                title=self.cov_cols[i], unit=self._bev_units[i], cmap=self.cmap,
                norm=self._bev_norms[i], point_size=self.point_size,
                start_xy=start_xy, show_axes_labels=False,
                current_xy=cur_xy
            )

    def _refresh_bev_markers(self):
        try:
            self._show_bev_idx(int(self.selected_cov_idx))
            self._redraw_thumbnails_with_markers()
        except Exception:
            pass
    @QtCore.pyqtSlot(int)
    def _show_bev_idx(self, idx: int):
        if idx < 0 or idx >= len(self.cov_cols):
            return
        self.main_canvas.reset_hover()
        self.selected_cov_idx = int(idx)

        start_xy = (self._bev_x[0], self._bev_y[0]) if self._bev_x.size > 0 else None
        cur_xy = self._current_bev_dot_xy()
        self.main_canvas.draw_scatter(
            self._bev_x, self._bev_y, self._bev_ci_vals[idx],
            title=self.cov_cols[idx], unit=self._bev_units[idx],
            cmap=self.cmap, norm=self._bev_norms[idx], point_size=self.point_size,
            start_xy=start_xy, show_axes_labels=True,
            current_xy=cur_xy
        )
        t_vals = self._bev_t if self._bev_t is not None and len(self._bev_t) == len(self._bev_x) else np.arange(len(self._bev_x), dtype=float)
        self.main_canvas.set_hover_data(
            self._bev_x, self._bev_y, t_vals, self._bev_ci_vals[idx], self._bev_units[idx], self.cov_cols[idx],
            lidar_index_resolver=self._lidar_index_for_gps_time
        )
        if self.centerStack.currentIndex() == 0:
            self._update_pointcloud(force=True)
        # thumbnails are redrawn by _refresh_bev_markers when index/offset change

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
        self._update_images()

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
        self._update_images()
        
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

    def _on_thumb_clicked(self, cam_1based: int):
        # update selected camera and refresh images (no hover reaction)
        try:
            self._selected_cam_1based = int(cam_1based)
        except Exception:
            self._selected_cam_1based = 1
        if hasattr(self, "imagePanel") and self.imagePanel is not None:
            self.imagePanel.set_selected_cam(self._selected_cam_1based)
        # redraw main image + thumbnails using current LiDAR time and per-cam offsets
        if hasattr(self, "_update_images"):
            self._update_images()

    # ---- Camera images ----
    def _parse_cam_timestamp(self, filename: str) -> Optional[float]:
        # Expect: cam{n}_frame_{idx}_{sec}_{nanosec}.jpg
        try:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)
            parts = name.split("_")
            # e.g., ['cam1', 'frame', '000000', '1755700589', '358076076']
            sec = int(parts[-2])
            nsec = int(parts[-1])
            return float(sec) + 1e-9 * float(nsec)
        except Exception:
            return None

    def _load_camera_set(self, root_dir: str):
        # Validate folders
        cams = []
        times = []
        for i in range(1, 7):
            d = os.path.join(root_dir, f"camera_{i}")
            if not os.path.isdir(d):
                raise RuntimeError(f"Missing folder: camera_{i}")
            files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg")])
            if len(files) == 0:
                raise RuntimeError(f"No JPGs in {d}")
            t = []
            for fp in files:
                tv = self._parse_cam_timestamp(fp)
                if tv is None:
                    raise RuntimeError(f"Failed to parse timestamp: {fp}")
                t.append(tv)
            cams.append(files)
            times.append(np.array(t, dtype=np.float64))
        self.cam_files = cams
        self.cam_times = times
        self.cam_loaded_root = root_dir
        self._selected_cam_1based = 1
        # Initial draw: if LiDAR known use that time, else first frames
        self._update_images(force_first_if_no_lidar=True)

    def _get_pixmap_cached(self, path: str) -> Optional['QtGui.QPixmap']:
        try:
            from PyQt5 import QtGui as _QtGui
        except Exception:
            return None
        pm = self._pix_cache.get(path)
        if pm is None or pm.isNull():
            pm = _QtGui.QPixmap(path)
            self._pix_cache[path] = pm
        return pm

    def _update_images(self, force_first_if_no_lidar: bool = False):
        # Update main image and thumbnails based on LiDAR time and per-cam offsets
        if self.cam_files is None or self.cam_times is None:
            return
        # Determine reference time
        ref_t = None
        if self.scans is not None and len(self.scans) > 0:
            idx = int(np.clip(self.current_index, 0, len(self.scans) - 1))
            ref_t = float(self.scans[idx].t)
        elif force_first_if_no_lidar:
            # Use first frame time of cam1
            ref_t = float(self.cam_times[0][0])
        if ref_t is None:
            return
        # Select frame per camera using nearest to (ref_t + offset_i)
        thumbs = {}
        sel_pm = None
        sel_cam = int(self._selected_cam_1based)
        for cam1 in range(1, 7):
            t_arr = self.cam_times[cam1 - 1]
            offs = float(self.cam_offsets_ms[cam1 - 1]) * 1e-3
            tgt = ref_t + offs
            j = int(np.argmin(np.abs(t_arr - tgt)))
            path = self.cam_files[cam1 - 1][j]
            pm = self._get_pixmap_cached(path)
            if pm:
                thumbs[cam1] = pm
            if cam1 == sel_cam:
                sel_pm = pm
        # Apply to UI
        self.imagePanel.set_thumbnails(thumbs)
        self.imagePanel.set_selected_cam(sel_cam)
        self.imagePanel.set_main_pixmap(sel_pm)


class AspectImageView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: Optional[QtGui.QPixmap] = None
        self.setMinimumHeight(120)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor("#1e1f22"))
        self.setPalette(pal)

    def set_pixmap(self, pix: Optional[QtGui.QPixmap]):
        self._pix = pix
        self.update()

    def paintEvent(self, ev: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1e1f22"))
        if not self._pix or self._pix.isNull():
            return
        w = self.width()
        h = self.height()
        img_w = self._pix.width()
        img_h = self._pix.height()
        if img_w <= 0 or img_h <= 0:
            return
        # contain: preserve aspect, one dimension fits, letterbox allowed
        r_widget = w / float(max(1, h))
        r_img = img_w / float(max(1, img_h))
        if r_img >= r_widget:
            # fit width
            draw_w = w
            draw_h = int(round(w / r_img))
        else:
            # fit height
            draw_h = h
            draw_w = int(round(h * r_img))
        x = int((w - draw_w) * 0.5)
        y = int((h - draw_h) * 0.5)
        target = QtCore.QRect(x, y, draw_w, draw_h)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p.drawPixmap(target, self._pix)
        p.end()


class CoverImageView(QtWidgets.QWidget):
    # Draws pixmap with aspect ratio preserved but covering the entire widget (center-crop).
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix: Optional[QtGui.QPixmap] = None
        self.setMinimumSize(80, 60)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor("#1e1f22"))
        self.setPalette(pal)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_pixmap(self, pix: Optional[QtGui.QPixmap]):
        self._pix = pix
        self.update()

    def paintEvent(self, ev: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1e1f22"))
        if not self._pix or self._pix.isNull():
            return
        w = self.width()
        h = self.height()
        iw = self._pix.width()
        ih = self._pix.height()
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            return
        # cover: scale so that both dimensions are >= widget; then center
        sx = w / float(iw)
        sy = h / float(ih)
        s = max(sx, sy)
        draw_w = int(round(iw * s))
        draw_h = int(round(ih * s))
        x = int((w - draw_w) * 0.5)
        y = int((h - draw_h) * 0.5)
        target = QtCore.QRect(x, y, draw_w, draw_h)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        p.drawPixmap(target, self._pix)
        p.end()


class ThumbnailItem(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(int)

    def __init__(self, cam_idx_1based: int, label_text: str, parent=None):
        super().__init__(parent)
        self._cam_idx = int(cam_idx_1based)
        self._selected = False

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(1)
        # Use constant 2px border in both states to avoid size changes on selection
        self.setStyleSheet("QFrame{border:2px solid #3b3f45; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")

        # Ensure uniform item geometry
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(160)
        self.setMaximumHeight(220)

        self.imgView = CoverImageView()
        self.textLabel = QtWidgets.QLabel(label_text)
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setMinimumHeight(18)
        self.textLabel.setMaximumHeight(22)
        self.textLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        v.addWidget(self.imgView, 1)
        v.addWidget(self.textLabel, 0)

    def set_selected(self, sel: bool):
        self._selected = bool(sel)
        if self._selected:
            # red border, constant width 2px
            self.setStyleSheet("QFrame{border:2px solid red; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")
        else:
            # grey border, constant width 2px
            self.setStyleSheet("QFrame{border:2px solid #3b3f45; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")

    def set_thumbnail(self, pix: Optional[QtGui.QPixmap]):
        self.imgView.set_pixmap(pix if pix and not pix.isNull() else None)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        self.clicked.emit(self._cam_idx)
        super().mousePressEvent(ev)


class ImagePanel(QtWidgets.QWidget):
    thumbnailClicked = QtCore.pyqtSignal(int)  # 1-based cam idx

    def __init__(self, parent=None):
        super().__init__(parent)
        # Vertical splitter: top main image, bottom 2x3 thumbnails
        self.vsplit = QtWidgets.QSplitter(Qt.Vertical)
        self.mainView = AspectImageView()
        self.thumbContainer = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.thumbContainer)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setSpacing(8)
        # Enforce equal column widths and row heights
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)

        # Fixed 2x3 order: [cam2, cam1, cam6; cam3, cam4, cam5]
        self._cam_order_1based: List[int] = [2, 1, 6, 3, 4, 5]
        self._labels: List[str] = ["Front-Left", "Front", "Front-Right", "Rear-Left", "Rear", "Rear-Right"]
        self._thumbs: List[ThumbnailItem] = []
        for i, cam1 in enumerate(self._cam_order_1based):
            ti = ThumbnailItem(cam1, self._labels[i])
            ti.clicked.connect(self.thumbnailClicked)
            self._thumbs.append(ti)
            r = 0 if i < 3 else 1
            c = i if i < 3 else i - 3
            self.grid.addWidget(ti, r, c)

        self.vsplit.addWidget(self.mainView)
        self.vsplit.addWidget(self.thumbContainer)
        self.thumbContainer.setMinimumHeight(180)
        self.thumbContainer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.vsplit, 1)

        self._selected_cam_1based = 1
        self._set_selected_cam_visuals()

    def set_labels(self, labels_in_order: List[str]):
        if len(labels_in_order) == 6:
            self._labels = list(labels_in_order)
            for i, lbl in enumerate(self._labels):
                self._thumbs[i].textLabel.setText(lbl)

    def set_cam_order(self, order_1based: List[int]):
        if len(order_1based) == 6 and sorted(order_1based) == [1, 2, 3, 4, 5, 6]:
            self._cam_order_1based = list(order_1based)
            for i, ti in enumerate(self._thumbs):
                r = 0 if i < 3 else 1
                c = i if i < 3 else i - 3
                self.grid.addWidget(ti, r, c)

    def set_main_pixmap(self, pix: Optional[QtGui.QPixmap]):
        self.mainView.set_pixmap(pix)

    def set_thumbnails(self, cam_idx_to_pix: Dict[int, QtGui.QPixmap]):
        for i, cam1 in enumerate(self._cam_order_1based):
            pix = cam_idx_to_pix.get(int(cam1))
            self._thumbs[i].set_thumbnail(pix)

    def set_selected_cam(self, cam_1based: int):
        self._selected_cam_1based = int(cam_1based)
        self._set_selected_cam_visuals()

    def _set_selected_cam_visuals(self):
        for i, cam1 in enumerate(self._cam_order_1based):
            self._thumbs[i].set_selected(cam1 == self._selected_cam_1based)

    def selected_cam(self) -> int:
        return int(self._selected_cam_1based)


def main():
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1280, 900)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
