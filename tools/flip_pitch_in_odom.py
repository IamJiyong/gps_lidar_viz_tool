#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def flip_pitch_in_quaternions(df: pd.DataFrame, cols):
    q = df[list(cols)].to_numpy(dtype=np.float64)

    e_xyz = R.from_quat(q).as_euler("xyz", degrees=False)
    e_xyz[:, 1] *= -1.0  # flip pitch
    q_new = R.from_euler("xyz", e_xyz, degrees=False).as_quat()

    norms = np.linalg.norm(q_new, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_new = q_new / norms

    return q_new


def summarize_pitch(q_orig: np.ndarray, q_new: np.ndarray):
    pitch_orig = R.from_quat(q_orig).as_euler("xyz", degrees=True)[:, 1]
    pitch_new = R.from_quat(q_new).as_euler("xyz", degrees=True)[:, 1]
    return {
        "orig_pitch_mean_deg": float(np.mean(pitch_orig)),
        "new_pitch_mean_deg": float(np.mean(pitch_new)),
        "orig_pitch_min_deg": float(np.min(pitch_orig)),
        "new_pitch_min_deg": float(np.min(pitch_new)),
        "orig_pitch_max_deg": float(np.max(pitch_orig)),
        "new_pitch_max_deg": float(np.max(pitch_new)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Flip pitch sign in orientation quaternions (x,y,z,w) and write a new CSV."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_pitch_flipped{in_path.suffix}")
    cols = ["ori_x", "ori_y", "ori_z", "ori_w"]

    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    for c in cols:
        if c not in df.columns:
            print(f"Missing column in CSV: {c}", file=sys.stderr)
            sys.exit(1)

    q_orig = df[list(cols)].to_numpy(dtype=np.float64)
    q_new = flip_pitch_in_quaternions(df, cols)

    df_out = df.copy()
    df_out[list(cols)] = q_new
    try:
        df_out.to_csv(out_path, index=False)
    except Exception as e:
        print(f"Failed to write CSV: {e}", file=sys.stderr)
        sys.exit(1)

    summary = summarize_pitch(q_orig, q_new)
    result = {
        "output": str(out_path),
        "rows": int(len(df)),
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()