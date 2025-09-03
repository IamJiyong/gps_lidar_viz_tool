from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Pose:
    t: float
    position: np.ndarray
    quaternion_xyzw: np.ndarray


def _normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return q / norms


def _enforce_quaternion_sign_continuity(q: np.ndarray) -> np.ndarray:
    if q.shape[0] == 0:
        return q
    q_out = q.copy()
    for i in range(1, q_out.shape[0]):
        if np.dot(q_out[i - 1], q_out[i]) < 0.0:
            q_out[i] = -q_out[i]
    return q_out


def pose_to_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(quaternion_xyzw)
    rot_mat = rot.as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = position
    # change T only yaw
    T[:3, :3] = rot_mat

    return T


@dataclass
class PoseInterpolators:
    pos_interp_x: Callable[[np.ndarray], np.ndarray]
    pos_interp_y: Callable[[np.ndarray], np.ndarray]
    pos_interp_z: Callable[[np.ndarray], np.ndarray]
    slerp: Slerp
    t_min: float
    t_max: float


def build_interpolators(df: pd.DataFrame, verbose: bool) -> PoseInterpolators:
    t = df["t"].to_numpy(dtype=np.float64)
    p = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    q = df[["ori_x", "ori_y", "ori_z", "ori_w"]].to_numpy(dtype=np.float64)
    q = _normalize_quaternions(q)
    q = _enforce_quaternion_sign_continuity(q)

    # allow extrapolation for positions
    pos_interp_x = interp1d(t, p[:, 0], kind="linear", bounds_error=False, fill_value="extrapolate")
    pos_interp_y = interp1d(t, p[:, 1], kind="linear", bounds_error=False, fill_value="extrapolate")
    pos_interp_z = interp1d(t, p[:, 2], kind="linear", bounds_error=False, fill_value="extrapolate")

    rot = Rotation.from_quat(q)
    slerp = Slerp(t, rot)

    return PoseInterpolators(pos_interp_x, pos_interp_y, pos_interp_z, slerp, float(t[0]), float(t[-1]))


@dataclass
class ResampledPoses:
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    transforms: np.ndarray


def resample_poses(df: pd.DataFrame, target_rate_hz: float, verbose: bool) -> ResampledPoses:
    t_min = df["t"].iloc[0]
    t_max = df["t"].iloc[-1]
    if target_rate_hz <= 0:
        raise ValueError("target_rate must be > 0")
    dt = 1.0 / target_rate_hz
    # Build time grid that never exceeds t_max
    num = int(np.floor((t_max - t_min) / dt)) + 1
    num = max(1, num)
    times = t_min + np.arange(num, dtype=np.float64) * dt
    # Numerical safety: clamp any tiny overshoot
    if times[-1] > t_max:
        times[-1] = t_max

    interps = build_interpolators(df, verbose=False)

    px = interps.pos_interp_x(times)
    py = interps.pos_interp_y(times)
    pz = interps.pos_interp_z(times)
    positions = np.stack([px, py, pz], axis=1)

    rots = interps.slerp(times)
    quaternions = rots.as_quat()
    quaternions = _normalize_quaternions(quaternions)
    quaternions = _enforce_quaternion_sign_continuity(quaternions)

    transforms = np.zeros((len(times), 4, 4), dtype=np.float64)
    for i in range(len(times)):
        transforms[i] = pose_to_matrix(positions[i], quaternions[i])

    return ResampledPoses(times, positions, quaternions, transforms)


def evaluate_pose(interps: PoseInterpolators, t_query: float) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array(
        [
            float(interps.pos_interp_x([t_query])[0]),
            float(interps.pos_interp_y([t_query])[0]),
            float(interps.pos_interp_z([t_query])[0]),
        ]
    )
    # clamp for orientation slerp (Slerp doesn't extrapolate)
    tq_rot = float(np.clip(t_query, interps.t_min, interps.t_max))
    rot = interps.slerp([tq_rot])
    q = rot.as_quat()[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    return p, q


def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4)
    if points_xyz.size == 0:
        return points_xyz
    pts_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1
    )
    pts_out = (T @ pts_h.T).T[:, :3]
    return pts_out
    
def update_heading_from_path(
    df: pd.DataFrame,
    min_speed: float = 1e-6,     # '속도'가 아니라 diff_stride 간격의 3D 이동량 임계값
    diff_stride: int = 10,        # 차분에 사용할 간격(샘플 수)
    verbose: bool = False,
) -> pd.DataFrame:
    """
    XY 경로로 yaw(heading), 3D 경로 기울기로 pitch를 계산해
    ori_x, ori_y, ori_z, ori_w를 (roll=0, pitch, yaw)로 덮어쓴다.
    또한 'heading'[rad], 'pitch'[rad] 컬럼을 추가한다.

    차분은 중앙차분을 기본으로 하되, 양 끝단은 전/후진 차분을 사용한다.
    차분 간격은 diff_stride(기본 5)로 지정한다.

    가정:
      - base_link의 +X가 진행방향(앞)이며, world의 +Z가 '위'.
      - yaw = atan2(dy, dx)
      - pitch = atan2(dz, hypot(dx, dy)) (양의 값이면 코가 위로)
      - roll은 0으로 고정
    """
    required = {"pos_x", "pos_y", "pos_z"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain columns: {sorted(required)}")

    s = int(max(1, diff_stride))

    x = df["pos_x"].to_numpy(dtype=np.float64)
    y = df["pos_y"].to_numpy(dtype=np.float64)
    z = df["pos_z"].to_numpy(dtype=np.float64)
    n = x.size

    if n < 2:
        raise ValueError("Need at least 2 position samples to compute orientation.")
    if n <= s:
        raise ValueError(f"Need at least diff_stride+1 samples (got n={n}, diff_stride={s}).")

    # s-간격 중앙차분(내부), 전/후진 차분(가)
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)

    for i in range(n):
        if (i - s) >= 0 and (i + s) < n:
            # 중앙차분: x[i+s] - x[i-s]
            dx[i] = 0.5 * (x[i + s] - x[i - s])
            dy[i] = 0.5 * (y[i + s] - y[i - s])
            dz[i] = 0.5 * (z[i + s] - z[i - s])
        elif (i + s) < n:
            # 전진차분: x[i+s] - x[i]
            dx[i] = x[i + s] - x[i]
            dy[i] = y[i + s] - y[i]
            dz[i] = z[i + s] - z[i]
        else:
            # 후진차분: x[i] - x[i-s]
            dx[i] = x[i] - x[i - s]
            dy[i] = y[i] - y[i - s]
            dz[i] = z[i] - z[i - s]

    # Yaw(heading) & Pitch 계산
    yaw = np.arctan2(dy, dx)          # [-pi, pi]
    horiz = np.hypot(dx, dy)
    pitch = np.arctan2(dz, horiz)     # [-pi/2, pi/2]

    # 거의 정지한 샘플은 이전 각도 유지 (3D 이동량 기반)
    step_3d = np.sqrt(dx*dx + dy*dy + dz*dz)
    for i in range(1, n):
        if step_3d[i] < float(min_speed):
            yaw[i] = yaw[i - 1]
            pitch[i] = pitch[i - 1]

    # Yaw 연속화
    yaw = np.unwrap(yaw)

    # (roll=0, pitch, yaw) → 쿼터니언(x,y,z,w)
    rolls = np.zeros_like(yaw)
    angles = np.stack([yaw, pitch, rolls], axis=1)  # [yaw, pitch, roll]
    quats = Rotation.from_euler("zyx", angles, degrees=False).as_quat()

    df_out = df.copy()
    df_out["ori_x"] = quats[:, 0]
    df_out["ori_y"] = quats[:, 1]
    df_out["ori_z"] = quats[:, 2]
    df_out["ori_w"] = quats[:, 3]
    df_out["heading"] = yaw
    df_out["pitch"] = pitch

    if verbose:
        print(f"Recomputed yaw+pitch from path with diff_stride={s} (roll=0).", flush=True)
    return df_out


def update_heading_from_path_preserve_roll(
    df: pd.DataFrame,
    *,
    min_speed_mps: float = 0.05,
    step_epsilon: float = 1e-4,
    diff_stride: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Recompute heading (yaw) and pitch from ENU path deltas while PRESERVING the
    original per-sample roll. Produces updated quaternions (ori_x/y/z/w).

    - Yaw = atan2(dy, dx)
    - Pitch = atan2(dz, hypot(dx, dy))
    - Roll = original roll extracted from the input quaternion

    Motion gating:
      - If speed (sqrt(vel_x^2+vel_y^2+vel_z^2)) < min_speed_mps, keep original quaternion
      - If velocity not available, fall back to a tiny step threshold on the path deltas

    Boundary policy:
      - First/last samples keep original quaternion
    """
    required_pos = {"pos_x", "pos_y", "pos_z"}
    required_ori = {"ori_x", "ori_y", "ori_z", "ori_w"}
    if not required_pos.issubset(df.columns) or not required_ori.issubset(df.columns):
        raise ValueError(f"df must contain columns: {sorted(required_pos | required_ori)}")

    # Extract arrays
    x = df["pos_x"].to_numpy(dtype=np.float64)
    y = df["pos_y"].to_numpy(dtype=np.float64)
    z = df["pos_z"].to_numpy(dtype=np.float64)
    q0 = df[["ori_x", "ori_y", "ori_z", "ori_w"]].to_numpy(dtype=np.float64)
    q0 = _normalize_quaternions(q0)
    q0 = _enforce_quaternion_sign_continuity(q0)
    n = x.size
    if n < 2:
        return df.copy()

    # Original angles (for preserving roll, and fallback yaw/pitch)
    r0 = Rotation.from_quat(q0)
    # zyx: returns [yaw, pitch, roll]
    ang0 = r0.as_euler("zyx", degrees=False)
    yaw0 = ang0[:, 0]
    pitch0 = ang0[:, 1]
    roll0 = ang0[:, 2]

    s = int(max(1, diff_stride))
    dx = np.empty(n, dtype=np.float64)
    dy = np.empty(n, dtype=np.float64)
    dz = np.empty(n, dtype=np.float64)
    for i in range(n):
        if (i - s) >= 0 and (i + s) < n:
            dx[i] = 0.5 * (x[i + s] - x[i - s])
            dy[i] = 0.5 * (y[i + s] - y[i - s])
            dz[i] = 0.5 * (z[i + s] - z[i - s])
        elif (i + s) < n:
            dx[i] = x[i + s] - x[i]
            dy[i] = y[i + s] - y[i]
            dz[i] = z[i + s] - z[i]
        else:
            dx[i] = x[i] - x[i - s]
            dy[i] = y[i] - y[i - s]
            dz[i] = z[i] - z[i - s]

    yaw_new = np.arctan2(dy, dx)
    horiz = np.hypot(dx, dy)
    pitch_new = np.arctan2(dz, horiz)

    # Unwrap yaw for continuity
    yaw_new = np.unwrap(yaw_new)

    # Motion gating using velocity if available; else use step magnitude
    has_vel = all(c in df.columns for c in ("vel_x", "vel_y", "vel_z"))
    if has_vel:
        vx = df["vel_x"].to_numpy(dtype=np.float64)
        vy = df["vel_y"].to_numpy(dtype=np.float64)
        vz = df["vel_z"].to_numpy(dtype=np.float64)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        moving = speed >= float(min_speed_mps)
    else:
        step_3d = np.sqrt(dx * dx + dy * dy + dz * dz)
        moving = step_3d >= float(step_epsilon)

    # Boundary handling: force keep original for first/last
    if n >= 1:
        moving[0] = False
        moving[-1] = False

    # Blend yaw/pitch: new when moving else original
    yaw_final = np.where(moving, yaw_new, yaw0)
    pitch_final = np.where(moving, pitch_new, pitch0)
    roll_final = roll0  # always preserve original roll

    # Rebuild quaternion from (yaw, pitch, roll)
    angles = np.stack([yaw_final, pitch_final, roll_final], axis=1)
    q_new = Rotation.from_euler("zyx", angles, degrees=False).as_quat()

    df_out = df.copy()
    df_out["ori_x"] = q_new[:, 0]
    df_out["ori_y"] = q_new[:, 1]
    df_out["ori_z"] = q_new[:, 2]
    df_out["ori_w"] = q_new[:, 3]
    df_out["heading"] = yaw_final
    df_out["pitch"] = pitch_final

    if verbose:
        print(
            f"Recomputed yaw+pitch from path (preserve roll), diff_stride={s}, "
            f"min_speed_mps={min_speed_mps}, step_eps={step_epsilon}.",
            flush=True,
        )
    return df_out
