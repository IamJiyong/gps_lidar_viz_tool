#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
from matplotlib.colors import LogNorm, Normalize


# ---------- I/O & Geo ----------

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


def xy_from_df(df: pd.DataFrame, origin_lat: float | None, origin_lon: float | None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    has_geo = "latitude" in df.columns and "longitude" in df.columns
    has_local = "pos_x" in df.columns and "pos_y" in df.columns
    if not has_geo and not has_local:
        raise ValueError("CSV must contain either ('latitude','longitude') or ('pos_x','pos_y').")

    if has_geo:
        lat_series = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=float)
        lon_series = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(lat_series) & np.isfinite(lon_series)
        lat_series = lat_series[valid_mask]
        lon_series = lon_series[valid_mask]
        df = df.loc[valid_mask].reset_index(drop=True)
        lat0 = float(origin_lat) if origin_lat is not None else float(lat_series[0])
        lon0 = float(origin_lon) if origin_lon is not None else float(lon_series[0])
        transformer = build_local_transformer(lat0=lat0, lon0=lon0)
        x_m, y_m = transformer.transform(lon_series, lat_series)
    else:
        x_m = pd.to_numeric(df["pos_x"], errors="coerce").to_numpy(dtype=float)
        y_m = pd.to_numeric(df["pos_y"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(x_m) & np.isfinite(y_m)
        x_m = x_m[valid_mask]
        y_m = y_m[valid_mask]
        df = df.loc[valid_mask].reset_index(drop=True)

    return x_m, y_m, df


# ---------- Covariance -> 95% CI ----------

COV_COLS_DEFAULT = [
    "pos_cov_x", "pos_cov_y", "pos_cov_z",
    "pos_cov_pitch", "pos_cov_yaw", "pos_cov_roll",
]

@dataclass
class CIResult:
    name: str
    ci_values: np.ndarray  # color values
    unit: str              # "m" or "deg"


def compute_ci_arrays(df: pd.DataFrame, cov_cols: List[str] | None = None) -> List[CIResult]:
    """Convert covariance(variance) columns to 95% CI values per column.

    Assumptions:
      - Position covariance columns are in m^2 (output in meters)
      - Orientation covariance columns are in deg^2 (output in degrees)
    """
    Z95 = 1.959963984540054  # two-sided 95% for Normal
    cols = cov_cols or COV_COLS_DEFAULT

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required covariance columns: {missing}")

    results: List[CIResult] = []
    for name in cols:
        var = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        var = np.where(np.isfinite(var), var, np.nan)
        var = np.clip(var, 0.0, None)
        ci = Z95 * np.sqrt(var)
        if any(k in name for k in ("yaw", "pitch", "roll")):
            # input variance is deg^2 -> CI already in degrees
            unit = "deg"
        else:
            unit = "m"
        results.append(CIResult(name=name, ci_values=ci, unit=unit))
    return results


# ---------- Normalizations ----------

def make_log_pos_norm(vals: np.ndarray) -> LogNorm:
    vpos = vals[np.isfinite(vals) & (vals > 0.0)]
    if vpos.size == 0:
        return LogNorm(vmin=1.0, vmax=1.0)
    vmin = max(float(np.min(vpos)), 1e-12)
    vmax = max(float(np.max(vpos)), vmin * 1.000001)
    return LogNorm(vmin=vmin, vmax=vmax)


def build_norms(ci_results: List[CIResult], mode: str = "per"):
    """Return list of Norms aligned with ci_results ('per' or 'shared')."""
    if mode not in ("per", "shared"):
        mode = "per"

    if mode == "shared":
        all_vals = np.concatenate([r.ci_values[np.isfinite(r.ci_values) & (r.ci_values > 0.0)] for r in ci_results])
        shared = make_log_pos_norm(all_vals)
        return [shared] * len(ci_results)
    else:
        return [make_log_pos_norm(r.ci_values) for r in ci_results]


# ---------- Convenience ----------

def select_stride(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, int(k))
    return x[::k], y[::k]
