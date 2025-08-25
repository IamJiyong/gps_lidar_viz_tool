#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from matplotlib.patches import Ellipse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot BEV trajectory colored by covariance components from CSV (supports lat/lon or pos_x/pos_y).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="Path to data directory (e.g., /home/jiyong/Jiyong/Humzee/humzee_data)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Path to CSV (e.g., GPS_data.csv or Odom_data.csv)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every k-th sample for plotting (default: 1)",
    )
    parser.add_argument(
        "--origin_lat",
        type=float,
        default=None,
        help="Override origin latitude (deg). If not set, first sample latitude is used (only for lat/lon input).",
    )
    parser.add_argument(
        "--origin_lon",
        type=float,
        default=None,
        help="Override origin longitude (deg). If not set, first sample longitude is used (only for lat/lon input).",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["shared", "per"],
        default="per",
        help="Color normalization across subplots: 'shared' uses common range, 'per' normalizes each subplot independently.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=6.0,
        help="Scatter point size (default: 6.0)",
    )
    parser.add_argument(
        "--linthresh_frac",
        type=float,
        default=1e-3,
        help="SymLogNorm linear threshold as a fraction of max(|value|) when negatives exist (default: 1e-3).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path for the 2x3 covariance figure (e.g., trajectory_cov.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GPS Trajectory (BEV)",
        help="Overall figure title for covariance plots",
    )
    parser.add_argument(
        "--out_color",
        type=str,
        default=None,
        help="Output image path for the single color-by plot (e.g., trajectory_color_by.png).",
    )
    parser.add_argument(
        "--single_plot",
        action="store_true",
        help="Plot subplots for each covariance component.",
    )
    return parser


def build_local_transformer(lat0: float, lon0: float) -> Transformer:
    return Transformer.from_crs(
        "epsg:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84 +units=m +no_defs",
        always_xy=True,
    )


def load_gps_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower()
    alias = {"lat": "latitude", "lon": "longitude", "pos_conv_z": "pos_cov_z"}
    for k, v in alias.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


def _make_log_norm(vals: np.ndarray, linthresh_frac: float):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return LogNorm(vmin=1.0, vmax=1.0)
    has_nonpos = np.any(vals <= 0.0)
    if not has_nonpos:
        vpos = vals[vals > 0.0]
        vmin = float(np.min(vpos)) if vpos.size > 0 else 1e-12
        vmax = float(np.max(vpos)) if vpos.size > 0 else 1e-12
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.000001)
        return LogNorm(vmin=vmin, vmax=vmax)
    amax = float(np.max(np.abs(vals))) if vals.size > 0 else 1.0
    amax = max(amax, 1e-12)
    linthresh = max(1e-12, linthresh_frac * amax)
    return SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-amax, vmax=amax, base=10)


def _make_linear_norm(vals: np.ndarray) -> Normalize:
    vpos = vals[np.isfinite(vals) & (vals > 0.0)]
    if vpos.size == 0:
        return Normalize(vmin=1.0, vmax=1.0)
    vmin = float(np.min(vpos))
    vmax = float(np.max(vpos))
    return Normalize(vmin=vmin, vmax=vmax)


def _make_log_pos_norm(vals: np.ndarray) -> LogNorm:
    """Strict log scale (positives only) for the single color-by plot."""
    vpos = vals[np.isfinite(vals) & (vals > 0.0)]
    if vpos.size == 0:
        return LogNorm(vmin=1.0, vmax=1.0)
    vmin = max(float(np.min(vpos)), 1e-12)
    vmax = max(float(np.max(vpos)), vmin * 1.000001)
    return LogNorm(vmin=vmin, vmax=vmax)


def _plot_covariance_2x3(
    x_m: np.ndarray,
    y_m: np.ndarray,
    cov_arrays: List[np.ndarray],
    cov_cols: List[str],
    args
) -> None:
    """
    각 서브플롯의 색을 '분산 -> 95% 오차범위(1.96*sqrt(var))'로 매핑.
    위치 축은 m, 각도 축은 deg로 컬러바 단위를 표기.
    """
    Z95 = 1.959963984540054  # 2-sided 95% (Normal)

    # 1) 공분산(=분산) -> 95% 오차범위로 변환, 각 컬럼의 단위 결정
    ci_arrays: List[np.ndarray] = []
    units: List[str] = []
    for name, var_arr in zip(cov_cols, cov_arrays):
        var = np.asarray(var_arr, dtype=float)
        # NaN/inf 제거 및 음수 보호
        var = np.where(np.isfinite(var), var, np.nan)
        var = np.clip(var, 0.0, None)

        ci = Z95 * np.sqrt(var)  # 95% 오차범위 (1D)
        # 각도 컬럼은 rad -> deg
        if any(k in name for k in ("yaw", "pitch", "roll")):
            ci = np.degrees(ci)
            units.append("deg")
        else:
            units.append("m")

        ci_arrays.append(ci)

    # 2) 정규화 객체 구성 (공유/개별) - 양수 전용 로그 스케일
    if args.norm == "shared":
        all_vals = np.concatenate(
            [c[np.isfinite(c) & (c > 0.0)] for c in ci_arrays if c.size > 0]
        ) if len(ci_arrays) else np.array([])
        shared_norm = _make_log_pos_norm(all_vals)
        norms = [shared_norm] * len(ci_arrays)
    else:
        norms = [_make_log_pos_norm(c) for c in ci_arrays]

    # 3) 플롯
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, cov_name, ci_vals, unit) in enumerate(zip(axes, cov_cols, ci_arrays, units)):
        # LogNorm 안전을 위해 0을 작은 양수로 클리핑
        cplot = np.clip(ci_vals, 1e-12, None)

        sc = ax.scatter(
            x_m, y_m,
            c=cplot,
            s=args.point_size,
            cmap=args.cmap,
            norm=norms[i],
            linewidths=0,
            rasterized=True,
        )
        ax.plot(x_m, y_m, "-", color="k", linewidth=0.6, alpha=0.5)
        ax.set_title(cov_name)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.6)

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(f"95% CI ({unit})")

    for ax in axes[::4]:
        ax.set_ylabel("Y (m)")
    for ax in axes[-4:]:
        ax.set_xlabel("X (m)")

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")


def _plot_covariance_single(x_m: np.ndarray, y_m: np.ndarray, cov_arrays: List[np.ndarray], cov_cols: List[str], args) -> None:
    # column별 norm 매핑 (미정의 컬럼은 기존 자동 스케일로 대응)
    norms = {}
    for name, c in zip(cov_cols, cov_arrays):
        if name == "pos_cov_yaw": # Normal
            norms[name] = Normalize(vmin=c.min(), vmax=c.max())
        else:
            norms[name] = _make_log_norm(c[np.isfinite(c)], args.linthresh_frac)

    # 저장 경로 설정
    out_dir = None
    out_stem = None
    if args.out:
        out_path = Path(args.out)
        out_dir = out_path.parent
        out_stem = out_path.stem

    # 항목별 개별 Figure로 플롯
    for cov_name, cov_vals in zip(cov_cols, cov_arrays):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        sc = ax.scatter(
            x_m, y_m,
            c=cov_vals,
            s=args.point_size,
            cmap=args.cmap,
            norm=norms[cov_name],
            linewidths=0,
            rasterized=True,
        )
        ax.plot(x_m, y_m, "-", color="k", linewidth=0.6, alpha=0.5)
        ax.set_title(cov_name)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cov_name)

        fig.tight_layout()
        if out_dir is not None and out_stem is not None:
            save_path = out_dir / f"{out_stem}_{cov_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")


def main() -> int:
    args = build_arg_parser().parse_args()

    # Finde latest CSV in the parent directory
    if args.csv is None:
        assert args.data_dir is not None
        csv_files = list(Path(args.data_dir).glob("*/GPS/Odom_data.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {args.data_dir}/GPS")
        args.csv = str(sorted(csv_files)[-1])

    if args.out is None:
        args.out = Path(args.csv).parent / "trajectory_var_bev.png"

    df = load_gps_csv(args.csv)
    has_geo = "latitude" in df.columns and "longitude" in df.columns
    has_local = "pos_x" in df.columns and "pos_y" in df.columns
    if not has_geo and not has_local:
        raise ValueError("CSV must contain either ('latitude','longitude') or ('pos_x','pos_y').")

    cov_cols: List[str] = [
        "pos_cov_x", "pos_cov_y", "pos_cov_z",
        "pos_cov_pitch", "pos_cov_yaw", "pos_cov_roll",
    ]
    missing_cov = [c for c in cov_cols if c not in df.columns]
    if missing_cov:
        raise ValueError(f"CSV missing required covariance columns: {missing_cov}")

    # XY in meters
    if has_geo:
        lat_series = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=float)
        lon_series = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(lat_series) & np.isfinite(lon_series)
        lat_series = lat_series[valid_mask]
        lon_series = lon_series[valid_mask]
        df = df.loc[valid_mask].reset_index(drop=True)
        lat0 = float(args.origin_lat) if args.origin_lat is not None else float(lat_series[0])
        lon0 = float(args.origin_lon) if args.origin_lon is not None else float(lon_series[0])
        transformer = build_local_transformer(lat0=lat0, lon0=lon0)
        x_m, y_m = transformer.transform(lon_series, lat_series)
    else:
        x_m = pd.to_numeric(df["pos_x"], errors="coerce").to_numpy(dtype=float)
        y_m = pd.to_numeric(df["pos_y"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(x_m) & np.isfinite(y_m)
        x_m = x_m[valid_mask]
        y_m = y_m[valid_mask]
        df = df.loc[valid_mask].reset_index(drop=True)

    # Covariance arrays
    cov_arrays = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cov_cols]

    # Apply stride
    stride = max(1, int(args.stride))
    x_plot = x_m[::stride]
    y_plot = y_m[::stride]

    _plot_covariance_2x3(x_plot, y_plot, cov_arrays, cov_cols, args)

    # Show all
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
