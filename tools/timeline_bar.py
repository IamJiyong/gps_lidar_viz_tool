# tools/timeline_bar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


@dataclass
class Band:
    id: str
    start_idx: int
    end_idx: int


class TimelineBar(QtWidgets.QWidget):
    currentIndexChanged = QtCore.pyqtSignal(int)
    bandsEdited = QtCore.pyqtSignal(object)  # List[Band]
    selectionChanged = QtCore.pyqtSignal(object)  # Set[str]
    removeRequested = QtCore.pyqtSignal(object)  # Set[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self._min_idx = 0
        self._max_idx = 0
        self._current = 0
        self._bands: List[Band] = []
        self._selected: Set[str] = set()
        self._dragging_edge: Optional[Tuple[str, str]] = None  # (band_id, 'start'|'end')
        self._dragging_cursor: bool = False
        self._font_small = QtGui.QFont(); self._font_small.setPointSize(8)
        # pending range (spacebar)
        self._pending: Optional[Tuple[int, int]] = None
        # context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context)

    # -------- API --------
    def set_range(self, min_idx: int, max_idx: int) -> None:
        self._min_idx = int(min_idx); self._max_idx = int(max_idx)
        self._current = max(self._min_idx, min(self._current, self._max_idx))
        self.update()

    def set_current(self, idx: int) -> None:
        v = max(self._min_idx, min(int(idx), self._max_idx))
        if v != self._current:
            self._current = v
            self.update()

    def set_bands(self, items: List[Band]) -> None:
        self._bands = list(items)
        self.update()

    def selected_ids(self) -> Set[str]:
        return set(self._selected)

    def set_pending_range(self, a: int, b: int) -> None:
        self._pending = (int(a), int(b))
        self.update()

    def clear_pending_range(self) -> None:
        self._pending = None
        self.update()

    # -------- Helpers --------
    def _x_from_idx(self, idx: int) -> int:
        if self._max_idx <= self._min_idx:
            return 0
        frac = (float(idx) - float(self._min_idx)) / float(self._max_idx - self._min_idx)
        return int(round(frac * (self.width() - 1)))

    def _idx_from_x(self, x: int) -> int:
        x = max(0, min(int(x), max(0, self.width() - 1)))
        if self.width() <= 1 or self._max_idx <= self._min_idx:
            return self._min_idx
        frac = float(x) / float(self.width() - 1)
        return int(round(self._min_idx + frac * (self._max_idx - self._min_idx)))

    # -------- Painting --------
    def paintEvent(self, ev: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1e1f22"))
        w = self.width(); h = self.height()
        # base track
        track_h = 20
        track_y = h - track_h - 28  # raise track to leave room for bottom labels
        p.fillRect(2, track_y, w - 4, track_h, QtGui.QColor("#2b2d31"))
        # pending range preview
        if self._pending is not None:
            a, b = self._pending
            x0 = self._x_from_idx(min(a, b)); x1 = self._x_from_idx(max(a, b))
            p.fillRect(QtCore.QRect(x0, track_y, max(1, x1 - x0 + 1), track_h), QtGui.QColor(255, 80, 80, 100))
        # bands
        for b in self._bands:
            x0 = self._x_from_idx(b.start_idx)
            x1 = self._x_from_idx(b.end_idx)
            if x1 < x0:
                x0, x1 = x1, x0
            r = QtCore.QRect(x0, track_y, max(3, x1 - x0 + 1), track_h)
            color = QtGui.QColor(220, 80, 80, 180)
            if b.id in self._selected:
                color = QtGui.QColor(255, 120, 120, 220)
            p.fillRect(r, color)
            # edges (wider grip)
            p.fillRect(QtCore.QRect(r.left(), r.top(), 6, r.height()), QtGui.QColor("#ffdddd"))
            p.fillRect(QtCore.QRect(r.right()-5, r.top(), 6, r.height()), QtGui.QColor("#ffdddd"))
            # labels: start above, end below
            p.setFont(self._font_small)
            p.setPen(QtGui.QColor("#e6e6e6"))
            p.drawText(x0+2, track_y-4, str(b.start_idx))
            p.drawText(x1-30, track_y+track_h+12, str(b.end_idx))
        # current cursor and label above it
        cx = self._x_from_idx(self._current)
        p.setFont(self._font_small)
        p.setPen(QtGui.QColor("#e6e6e6"))
        txt = str(self._current)
        fm = QtGui.QFontMetrics(p.font())
        tw = fm.width(txt)
        # clamp label Y to stay inside widget
        y_text = max(14, track_y - 34)
        # draw cursor line starting just below the text area
        p.setPen(QtGui.QPen(QtGui.QColor("#e6e6e6"), 2))
        y_line_top = max(y_text + 4, 2)
        p.drawLine(cx, y_line_top, cx, track_y + track_h)
        # draw text on top
        p.setPen(QtGui.QColor("#e6e6e6"))
        p.drawText(cx - tw//2, y_text, txt)
        p.end()

    # -------- Mouse --------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == Qt.LeftButton:
            hit = self._hit_test(ev.pos())
            if hit and hit[0] == "cursor_label":
                # try dark-themed int input if parent provides it
                val = None
                try:
                    parent = self.parent()
                    if parent and hasattr(parent, "_dark_int_input"):
                        val = parent._dark_int_input("Edit current index", "Index:", self._current, self._min_idx, self._max_idx)
                except Exception:
                    val = None
                if val is None:
                    val, ok = QtWidgets.QInputDialog.getInt(self, "Edit index", "Set index:", value=self._current, min=self._min_idx, max=self._max_idx)
                    if not ok:
                        return
                self._current = int(val)
                self.update(); self.currentIndexChanged.emit(self._current)
                return
            if hit and hit[0] == "edge":
                self._dragging_edge = (hit[1], hit[2])  # (id, which)
                self.setCursor(Qt.SplitHCursor)
                return
            elif hit and hit[0] == "band":
                bid = hit[1]
                if not (ev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self._selected.clear()
                if bid in self._selected:
                    self._selected.remove(bid)
                else:
                    self._selected.add(bid)
                self.selectionChanged.emit(self.selected_ids())
                self.update()
                return
            else:
                cx = self._x_from_idx(self._current)
                if abs(ev.x() - cx) <= 6:
                    self._dragging_cursor = True
                    return
                self._dragging_cursor = True
                self._current = self._idx_from_x(ev.x())
                self.update()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._dragging_edge:
            bid, which = self._dragging_edge
            idx = self._idx_from_x(ev.x())
            new_bands: List[Band] = []
            for b in self._bands:
                if b.id == bid:
                    if which == 'start':
                        b = Band(b.id, min(idx, b.end_idx), b.end_idx)
                    else:
                        b = Band(b.id, b.start_idx, max(idx, b.start_idx))
                new_bands.append(b)
            self._bands = new_bands
            self.bandsEdited.emit(self._bands)
            self.update()
        elif self._dragging_cursor:
            self._current = self._idx_from_x(ev.x())
            self.update()
        else:
            hit = self._hit_test(ev.pos())
            if hit and hit[0] == "edge":
                self.setCursor(Qt.SplitHCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == Qt.LeftButton:
            if self._dragging_edge:
                self._dragging_edge = None
                self.setCursor(Qt.ArrowCursor)
            if self._dragging_cursor:
                self._dragging_cursor = False
                self.currentIndexChanged.emit(self._current)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        hit = self._hit_test(ev.pos())
        if hit and hit[0] == "edge":
            bid, which = hit[1], hit[2]
            for i, b in enumerate(self._bands):
                if b.id != bid:
                    continue
                # dark-styled edit dialog
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Edit index")
                dlg.setStyleSheet("QDialog{background:#000; color:#fff;} QLabel{color:#fff;} QSpinBox{background:#1f2023; color:#e6e6e6; border:1px solid #3b3f45;} QPushButton{background:#2b2d31; color:#e6e6e6;}")
                form = QtWidgets.QFormLayout(dlg)
                label = QtWidgets.QLabel(f"Set {which} index:")
                box = QtWidgets.QSpinBox(); box.setRange(self._min_idx, self._max_idx); box.setValue(b.start_idx if which=='start' else b.end_idx)
                form.addRow(label)
                form.addRow(box)
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                form.addWidget(btns)
                btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    val = int(box.value())
                    if which == 'start':
                        self._bands[i] = Band(b.id, min(val, b.end_idx), b.end_idx)
                    else:
                        self._bands[i] = Band(b.id, b.start_idx, max(val, b.start_idx))
                    self.bandsEdited.emit(self._bands)
                    self.update()
                break
        super().mouseDoubleClickEvent(ev)

    def _on_context(self, pos: QtCore.QPoint):
        hit = self._hit_test(pos)
        if not hit or hit[0] == "cursor_label":
            return
        ids = list(self._selected)
        if hit[0] == "band" and hit[1] not in self._selected:
            ids = [hit[1]]
        menu = QtWidgets.QMenu(self)
        actEdit = None
        if len(ids) == 1:
            actEdit = menu.addAction("Edit…")
        actRemove = menu.addAction("Remove")
        act = menu.exec_(self.mapToGlobal(pos))
        if act is None:
            return
        if act is actRemove:
            self.removeRequested.emit(set(ids))
            return
        if act is actEdit and len(ids) == 1:
            bid = ids[0]
            for i, b in enumerate(self._bands):
                if b.id != bid:
                    continue
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Edit interval")
                dlg.setStyleSheet("QDialog{background:#000; color:#fff;} QLabel{color:#fff;} QSpinBox{background:#1f2023; color:#e6e6e6; border:1px solid #3b3f45;} QPushButton{background:#2b2d31; color:#e6e6e6;}")
                form = QtWidgets.QFormLayout(dlg)
                sBox = QtWidgets.QSpinBox(); sBox.setRange(self._min_idx, self._max_idx); sBox.setValue(b.start_idx)
                eBox = QtWidgets.QSpinBox(); eBox.setRange(self._min_idx, self._max_idx); eBox.setValue(b.end_idx)
                form.addRow("Start index", sBox)
                form.addRow("End index", eBox)
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                form.addWidget(btns)
                btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
                if dlg.exec_() == QtWidgets.QDialog.Accepted:
                    ns = int(min(sBox.value(), eBox.value()))
                    ne = int(max(sBox.value(), eBox.value()))
                    self._bands[i] = Band(b.id, ns, ne)
                    self.bandsEdited.emit(self._bands)
                    self.update()
                break

    # -------- Hit testing --------
    def _hit_test(self, pos: QtCore.QPoint) -> Optional[Tuple[str, str, str]]:
        w = self.width(); h = self.height()
        track_h = 20
        track_y = h - track_h - 28
        cx = self._x_from_idx(self._current)
        # use same clamped Y as paint
        y_text = max(14, track_y - 34)
        if QtCore.QRect(cx-18, y_text-10, 36, 20).contains(pos):
            return ("cursor_label", "", "")
        if pos.y() < track_y-30 or pos.y() > track_y + track_h + 30:
            return None
        for b in self._bands:
            x0 = self._x_from_idx(b.start_idx)
            x1 = self._x_from_idx(b.end_idx)
            if x1 < x0:
                x0, x1 = x1, x0
            r = QtCore.QRect(x0, track_y, max(3, x1 - x0 + 1), track_h)
            rHit = r.adjusted(-6, -6, +6, +6)
            if rHit.contains(pos):
                if abs(pos.x() - r.left()) <= 6:
                    return ("edge", b.id, 'start')
                if abs(pos.x() - r.right()) <= 6:
                    return ("edge", b.id, 'end')
                return ("band", b.id, '')
        return None