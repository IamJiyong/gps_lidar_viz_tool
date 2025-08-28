# tools/options_dialog.py
from PyQt5 import QtCore, QtWidgets


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