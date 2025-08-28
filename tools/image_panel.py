from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


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


class ThumbnailItem(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(int)

    def __init__(self, cam_idx_1based: int, label_text: str, parent=None):
        super().__init__(parent)
        self._cam_idx = int(cam_idx_1based)
        self._selected = False

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame{border:1px solid #3b3f45; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")

        self.imgLabel = QtWidgets.QLabel()
        self.imgLabel.setAlignment(Qt.AlignCenter)
        self.imgLabel.setMinimumSize(80, 60)
        self.textLabel = QtWidgets.QLabel(label_text)
        self.textLabel.setAlignment(Qt.AlignCenter)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        v.addWidget(self.imgLabel, 1)
        v.addWidget(self.textLabel, 0)

    def set_selected(self, sel: bool):
        self._selected = bool(sel)
        if self._selected:
            self.setStyleSheet("QFrame{border:2px solid red; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")
        else:
            self.setStyleSheet("QFrame{border:1px solid #3b3f45; background-color:#1e1f22;} QLabel{color:#e6e6e6;}")

    def set_thumbnail(self, pix: Optional[QtGui.QPixmap]):
        if pix and not pix.isNull():
            # scale down for label size while keeping aspect
            sz = self.imgLabel.size()
            if sz.width() <= 0 or sz.height() <= 0:
                scaled = pix.scaled(180, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaled = pix.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imgLabel.setPixmap(scaled)
        else:
            self.imgLabel.clear()

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

        # Create fixed 2x3 items in camera order mapping set by caller
        # Placeholder labels; caller will set actual labels
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
        # Give some sensible minimum for bottom area; leave constraints otherwise free
        self.thumbContainer.setMinimumHeight(160)

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
        # Order must be 6 unique camera indices from 1..6
        if len(order_1based) == 6 and sorted(order_1based) == [1, 2, 3, 4, 5, 6]:
            self._cam_order_1based = list(order_1based)
            # Recreate grid assignment while preserving widgets
            for i, ti in enumerate(self._thumbs):
                r = 0 if i < 3 else 1
                c = i if i < 3 else i - 3
                self.grid.addWidget(ti, r, c)

    def set_main_pixmap(self, pix: Optional[QtGui.QPixmap]):
        self.mainView.set_pixmap(pix)

    def set_thumbnails(self, cam_idx_to_pix: Dict[int, QtGui.QPixmap]):
        # cam_idx_to_pix keys are 1-based camera indices
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