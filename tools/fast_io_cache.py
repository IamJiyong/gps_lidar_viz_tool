# tools/fast_io_cache.py
from __future__ import annotations
import os
from typing import Iterable, Optional, Dict
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

__all__ = [
    "load_xyzi_file",
    "XYZIFileCache",
    "get_global_cache",
    "make_cached_loader",
    "infer_scan_path",
    "prefetch_ring",
    "try_hook_accumulate",
]

# ---------- 로우레벨 로더 ----------

def _normalize_xyzi(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    c = arr.shape[1]
    if c == 3:
        return np.c_[arr, np.zeros((arr.shape[0],), dtype=np.float32)]
    if c > 4:
        return arr[:, :4]
    if c < 4:
        pad = np.zeros((arr.shape[0], 4 - c), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)
    return arr

def load_xyzi_file(path: str) -> np.ndarray:
    """
    .bin(float32 packed), .npy(mmap), .npz(첫 배열) -> Nx4 float32 [x,y,z,i]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        return _normalize_xyzi(arr)
    if ext == ".npz":
        with np.load(path, mmap_mode="r") as z:
            arr = z[list(z.files)[0]]
            return _normalize_xyzi(arr)
    # default: .bin float32
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return raw.reshape(0, 4)
    n4 = raw.size // 4
    if n4 * 4 == raw.size:
        return raw.reshape(-1, 4)
    # xyz-only
    n3 = raw.size // 3
    xyz = raw[: n3 * 3].reshape(-1, 3).astype(np.float32, copy=False)
    i = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, i], axis=1)

# ---------- LRU + Prefetch ----------

class _LRU:
    def __init__(self, max_items: int = 64, max_bytes: Optional[int] = None):
        self.max_items = int(max_items)
        self.max_bytes = max_bytes
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    @staticmethod
    def _nbytes(a: np.ndarray) -> int:
        try:
            return int(a.nbytes)
        except Exception:
            return 0

    def get(self, k: str) -> Optional[np.ndarray]:
        with self._lock:
            v = self._d.pop(k, None)
            if v is not None:
                self._d[k] = v
            return v

    def put(self, k: str, v: np.ndarray) -> None:
        sz = self._nbytes(v)
        with self._lock:
            if k in self._d:
                self._bytes -= self._nbytes(self._d[k])
                self._d.pop(k, None)
            self._d[k] = v
            self._bytes += sz
            while (self.max_items is not None and len(self._d) > self.max_items) or \
                  (self.max_bytes is not None and self._bytes > self.max_bytes):
                kk, vv = self._d.popitem(last=False)
                self._bytes -= self._nbytes(vv)

class XYZIFileCache:
    """
    파일단 캐시 + 백그라운드 프리패치.
    - .load(path): 캐시에서 반환, 없으면 디스크에서 읽어 캐시 저장
    - .prefetch(paths): 비동기로 미리 읽어 OS 페이지캐시 + LRU 예열
    """
    def __init__(self, max_items: int = 64, max_bytes_mb: Optional[int] = None, workers: int = 4):
        self.cache = _LRU(max_items=max_items,
                          max_bytes=(max_bytes_mb * 1024 * 1024) if max_bytes_mb else None)
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._futs: Dict[str, "object"] = {}
        self._lock = Lock()

    def load(self, path: str) -> np.ndarray:
        if not path:
            return np.empty((0, 4), np.float32)
        arr = self.cache.get(path)
        if arr is not None:
            return arr
        with self._lock:
            fut = self._futs.get(path)
        if fut is not None:
            try:
                arr = fut.result()
            except Exception:
                arr = load_xyzi_file(path)
        else:
            arr = load_xyzi_file(path)
        self.cache.put(path, arr)
        return arr

    def prefetch(self, paths: Iterable[str]) -> None:
        with self._lock:
            for p in paths:
                if not p or p in self._futs:
                    continue
                if self.cache.get(p) is not None:
                    continue
                self._futs[p] = self.pool.submit(self._prefetch_one, p)

    def _prefetch_one(self, p: str) -> np.ndarray:
        try:
            a = load_xyzi_file(p)
            self.cache.put(p, a)
            return a
        except Exception:
            return np.empty((0, 4), np.float32)

# ---------- 전역 헬퍼 / 훅 ----------

_GLOBAL_CACHE: Optional[XYZIFileCache] = None

def get_global_cache(max_items: int = 64, max_bytes_mb: Optional[int] = None, workers: int = 4) -> XYZIFileCache:
    """전역 캐시 싱글턴을 돌려줍니다."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = XYZIFileCache(max_items=max_items, max_bytes_mb=max_bytes_mb, workers=workers)
    return _GLOBAL_CACHE

def make_cached_loader(cache: Optional[XYZIFileCache] = None):
    """accumulate 같은 곳에 전달할 수 있는 'path -> np.ndarray' 로더를 만듭니다."""
    c = cache or get_global_cache()
    def _loader(path: str) -> np.ndarray:
        return c.load(path)
    return _loader

# ---- 스캔 오브젝트에서 파일 경로 추정 ----

def infer_scan_path(scan, base_dir: Optional[str] = None) -> Optional[str]:
    """
    parse_lidar_directory()가 반환한 스캔 오브젝트에서 파일 경로를 최대한 유연하게 찾아냅니다.
    우선순위: .path / .filepath / .file / .filename / .fn -> 존재하면 그걸로.
    없으면 index 기반으로 base_dir에서 관례 파일명들을 탐색합니다.
    """
    # 1) 속성에서 직접 가져오기
    for attr in ("path", "filepath", "file", "filename", "fn"):
        p = getattr(scan, attr, None)
        if isinstance(p, str) and os.path.isfile(p):
            return p

    # 2) base_dir + index 패턴 추정
    if base_dir is None:
        return None
    try:
        idx = int(getattr(scan, "index"))
    except Exception:
        return None

    # 흔한 파일명 패턴들
    candidates = (
        f"{idx:06d}_xyzi.bin",
        f"{idx:06d}.bin",
        f"{idx:06d}.npy",
        f"{idx:06d}_xyzi.npy",
        f"{idx:06d}.npz",
    )
    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    return None

# ---- 중심 인덱스 주변 프레임 프리패치 ----

def prefetch_ring(scans, center_idx: int, radius: int = 3,
                  base_dir: Optional[str] = None,
                  cache: Optional[XYZIFileCache] = None) -> None:
    """
    현재 인덱스 주변 [center_idx - radius, center_idx + radius] 구간의 파일을 비동기로 예열합니다.
    accumulate 시 디스크 대기를 크게 줄여줍니다.
    """
    if scans is None:
        return
    n = len(scans)
    if n == 0 or radius <= 0:
        return
    lo = max(0, center_idx - int(radius))
    hi = min(n - 1, center_idx + int(radius))
    paths = []
    for i in range(lo, hi + 1):
        p = infer_scan_path(scans[i], base_dir=base_dir)
        if p:
            paths.append(p)
    if not paths:
        return
    (cache or get_global_cache()).prefetch(paths)

# ---- lidar_gnss.accumulate에 캐시 로더 주입(가능 시) ----

def try_hook_accumulate(cache: Optional[XYZIFileCache] = None) -> bool:
    """
    lidar_gnss.accumulate 모듈이 사용자 로더를 받을 수 있으면 캐시 로더를 연결합니다.
    - 지원되는 훅 이름(존재하는 경우에만): set_xyzi_loader, set_loader, XYZI_LOADER, POINT_LOADER, LOADER
    - 훅이 전혀 없으면 False를 돌려주며, 이 경우에도 prefetch만으로 체감 속도 개선 가능.
    """
    c = cache or get_global_cache()
    try:
        import lidar_gnss.accumulate as acc  # type: ignore
    except Exception:
        return False

    try:
        if hasattr(acc, "set_xyzi_loader"):
            acc.set_xyzi_loader(c.load)
            return True
        if hasattr(acc, "set_loader"):
            acc.set_loader(c.load)
            return True
        for name in ("XYZI_LOADER", "POINT_LOADER", "LOADER"):
            if hasattr(acc, name):
                setattr(acc, name, c.load)
                return True
    except Exception:
        return False
    return False