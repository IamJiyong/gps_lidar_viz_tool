# tools/options_dialog.py
from PyQt5 import QtCore, QtWidgets


class OptionsDialog(QtWidgets.QDialog):
    reloadRequested = QtCore.pyqtSignal()
    def __init__(self, parent=None, *, offset_min_ms=-1000, offset_max_ms=1000,
                 offset_slider_step_ms=1, marker_stride=5,
                 range_enabled=False, x_range=30.0, y_range=30.0, z_range=30.0,
                 max_points=0, max_frames=10, index_interval=1,
                 cam_offsets_ms=None,
                 worker_name: str = "",
                 save_root_dir: str = "",
                 color_mode: str = "default"):
        super().__init__(parent)
        self.setWindowTitle("Options")
        # Dark theme for dialog
        self.setStyleSheet("""
            QDialog { background-color: #1e1f22; color: #e6e6e6; }
            QLabel { color: #e6e6e6; }
            QGroupBox { border: 1px solid #3b3f45; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; color: #e6e6e6; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background-color: #1f2023; color: #e6e6e6; border: 1px solid #3b3f45; }
            QComboBox { background-color: #000000; color: #e6e6e6; border: 1px solid #3b3f45; }
            QComboBox QAbstractItemView { background-color: #000000; color: #e6e6e6; selection-background-color: #3b3f45; selection-color: #e6e6e6; }
            QCheckBox { color: #e6e6e6; }
            QDialogButtonBox { background-color: #2b2d31; }
            QPushButton { background-color: #2b2d31; color: #e6e6e6; border: 1px solid #3b3f45; padding: 4px 8px; }
            QPushButton:hover { background-color: #3b3f45; }
        """)

        if cam_offsets_ms is None or len(cam_offsets_ms) != 6:
            cam_offsets_ms = [100.0, 500.0, 500.0, 500.0, 500.0, 500.0]  # cam1..cam6

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

        # Camera offsets (ms)
        camBox = QtWidgets.QGroupBox("Camera offsets (ms)")
        camForm = QtWidgets.QFormLayout(camBox)
        self.camOffset1 = QtWidgets.QDoubleSpinBox(); self.camOffset1.setRange(-1e6, 1e6); self.camOffset1.setDecimals(1); self.camOffset1.setValue(float(cam_offsets_ms[0]))
        self.camOffset2 = QtWidgets.QDoubleSpinBox(); self.camOffset2.setRange(-1e6, 1e6); self.camOffset2.setDecimals(1); self.camOffset2.setValue(float(cam_offsets_ms[1]))
        self.camOffset3 = QtWidgets.QDoubleSpinBox(); self.camOffset3.setRange(-1e6, 1e6); self.camOffset3.setDecimals(1); self.camOffset3.setValue(float(cam_offsets_ms[2]))
        self.camOffset4 = QtWidgets.QDoubleSpinBox(); self.camOffset4.setRange(-1e6, 1e6); self.camOffset4.setDecimals(1); self.camOffset4.setValue(float(cam_offsets_ms[3]))
        self.camOffset5 = QtWidgets.QDoubleSpinBox(); self.camOffset5.setRange(-1e6, 1e6); self.camOffset5.setDecimals(1); self.camOffset5.setValue(float(cam_offsets_ms[4]))
        self.camOffset6 = QtWidgets.QDoubleSpinBox(); self.camOffset6.setRange(-1e6, 1e6); self.camOffset6.setDecimals(1); self.camOffset6.setValue(float(cam_offsets_ms[5]))
        # Labels use view mapping names
        camForm.addRow("Front (cam1)", self.camOffset1)
        camForm.addRow("Front-Left (cam2)", self.camOffset2)
        camForm.addRow("Rear-Left (cam3)", self.camOffset3)
        camForm.addRow("Rear (cam4)", self.camOffset4)
        camForm.addRow("Rear-Right (cam5)", self.camOffset5)
        camForm.addRow("Front-Right (cam6)", self.camOffset6)

        # Color mode
        self.colorMode = QtWidgets.QComboBox(); self.colorMode.addItems(["default", "intensity", "per-scan"])
        mode_list = ["default", "intensity", "per-scan"]
        try:
            idx = mode_list.index(color_mode)
            self.colorMode.setCurrentIndex(idx)
        except Exception:
            pass

        # Session options
        self.workerEdit = QtWidgets.QLineEdit(); self.workerEdit.setText(worker_name)
        self.saveRootEdit = QtWidgets.QLineEdit(); self.saveRootEdit.setText(save_root_dir)
        browseBtn = QtWidgets.QPushButton("Browse…")
        def _browse():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save root directory", self.saveRootEdit.text() or "")
            if d:
                self.saveRootEdit.setText(d)
        browseBtn.clicked.connect(_browse)
        saveRootRow = QtWidgets.QHBoxLayout(); saveRootRow.addWidget(self.saveRootEdit, 1); saveRootRow.addWidget(browseBtn, 0)

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
        form.addRow(camBox)
        form.addRow("Worker name", self.workerEdit)
        form.addRow("Default save root", saveRootRow)
        form.addRow("Point cloud color mode", self.colorMode)

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
            cam_offsets_ms=[
                float(self.camOffset1.value()),
                float(self.camOffset2.value()),
                float(self.camOffset3.value()),
                float(self.camOffset4.value()),
                float(self.camOffset5.value()),
                float(self.camOffset6.value()),
            ],
            worker_name=self.workerEdit.text().strip(),
            save_root_dir=self.saveRootEdit.text().strip(),
            color_mode=str(self.colorMode.currentText()),
        )