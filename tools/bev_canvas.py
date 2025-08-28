# tools/bev_canvas.py
import numpy as np
from typing import Optional, Tuple, Callable

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


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
                     show_axes_labels: bool = False,
                     current_xy: Optional[Tuple[float, float]] = None):
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

        # start marker (hollow red circle)
        if start_xy is not None:
            self.ax.scatter([start_xy[0]], [start_xy[1]],
                            s=120, facecolors="none", edgecolors="r", linewidths=2.0, zorder=5)

        # current index marker (filled red dot with white edge)
        if current_xy is not None:
            self.ax.scatter([current_xy[0]], [current_xy[1]],
                            s=60, facecolors="#ff3030", edgecolors="#ffffff",
                            linewidths=1.2, zorder=6)

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
    lidarIndexClicked = QtCore.pyqtSignal(int)  # emitted on click with resolved LiDAR index

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
        # Use Matplotlib's mouse press event (not Qt's) to access xdata/ydata reliably
        self.mpl_connect("button_press_event", self._on_button_press)

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

    def _on_button_press(self, event):
        try:
            if event.inaxes is self.ax and self._data_for_hover is not None:
                x_arr, y_arr, t_vals, _ci_vals, _unit, _title = self._data_for_hover
                if event.xdata is not None and event.ydata is not None and x_arr.size > 0:
                    cx, cy = float(event.xdata), float(event.ydata)
                    i = int(np.argmin((x_arr - cx) ** 2 + (y_arr - cy) ** 2))
                    t = float(t_vals[i])
                    if self._lidar_index_resolver is not None:
                        li = self._lidar_index_resolver(t)
                        if li is not None:
                            self.lidarIndexClicked.emit(int(li))
        except Exception:
            pass