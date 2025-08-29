# tools/marks_manager.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np


@dataclass
class Interval:
    start_idx: int
    end_idx: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "manual"  # "manual" or "auto"

    def normalized(self) -> "Interval":
        if self.start_idx <= self.end_idx:
            return self
        return Interval(self.end_idx, self.start_idx, id=self.id, source=self.source)


class MarksManager:
    def __init__(self, *, data_root: Optional[str] = None, worker: str = ""):
        self.data_root: Optional[str] = data_root
        self.worker: str = worker
        self.fixed_timestamp: Optional[str] = None  # YYYYMMDD_HHMMSS kept constant for this session file
        self.filename: Optional[str] = None
        self.manual: List[Interval] = []
        self.auto: List[Interval] = []
        # undo/redo history as list of (manual_copy, auto_copy)
        self._undo: List[Tuple[List[Interval], List[Interval]]] = []
        self._redo: List[Tuple[List[Interval], List[Interval]]] = []

    # ---------- Persistence ----------
    def _ensure_marks_dir(self, root: str) -> str:
        d = os.path.join(root, "marks_json")
        os.makedirs(d, exist_ok=True)
        return d

    def current_path(self) -> Optional[str]:
        return self.filename

    def set_worker(self, new_worker: str) -> None:
        new_worker = (new_worker or "").strip()
        if not new_worker:
            return
        if self.worker == new_worker:
            return
        old = self.current_path()
        self.worker = new_worker
        # move file keeping fixed timestamp
        if self.data_root and self.fixed_timestamp:
            new_path = self._build_new_filename(self.data_root, self.fixed_timestamp, self.worker)
            if old and os.path.isfile(old):
                try:
                    os.replace(old, new_path)
                except Exception:
                    # fallback: copy-then-remove semantics
                    try:
                        with open(old, "r") as f: txt = f.read()
                        with open(new_path, "w") as f: f.write(txt)
                        os.remove(old)
                    except Exception:
                        pass
            self.filename = new_path

    def set_save_root(self, root: str) -> None:
        self.data_root = root
        if self.fixed_timestamp and self.worker:
            self.filename = self._build_new_filename(root, self.fixed_timestamp, self.worker)

    def _build_new_filename(self, root: str, fixed_ts: str, worker: str) -> str:
        base = os.path.basename(os.path.normpath(root))
        marks_dir = self._ensure_marks_dir(root)
        name = f"gnss_{base}_{fixed_ts}_{worker}.json"
        return os.path.join(marks_dir, name)

    def _snapshot(self) -> Tuple[List[Interval], List[Interval]]:
        return ([Interval(iv.start_idx, iv.end_idx, id=iv.id, source=iv.source) for iv in self.manual],
                [Interval(iv.start_idx, iv.end_idx, id=iv.id, source=iv.source) for iv in self.auto])

    def _push_undo(self):
        self._undo.append(self._snapshot())
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        cur = self._snapshot()
        prev = self._undo.pop()
        self._redo.append(cur)
        self.manual, self.auto = prev
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cur = self._snapshot()
        nxt = self._redo.pop()
        self._undo.append(cur)
        self.manual, self.auto = nxt
        return True

    # ---------- Interval operations ----------
    @staticmethod
    def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
        if not intervals:
            return []
        items = [iv.normalized() for iv in intervals]
        items.sort(key=lambda x: (x.start_idx, x.end_idx))
        merged: List[Interval] = []
        cur = items[0]
        cur_ids: List[str] = [cur.id]
        same_source = True
        for iv in items[1:]:
            if iv.start_idx <= cur.end_idx + 1:
                cur.end_idx = max(cur.end_idx, iv.end_idx)
                cur_ids.append(iv.id)
                if iv.source != cur.source:
                    same_source = False
            else:
                preserved_id = cur_ids[0] if same_source and cur_ids and cur_ids[0] else uuid.uuid4().hex
                merged.append(Interval(cur.start_idx, cur.end_idx, id=preserved_id, source=cur.source))
                cur = iv
                cur_ids = [iv.id]
                same_source = True
        preserved_id = cur_ids[0] if same_source and cur_ids and cur_ids[0] else uuid.uuid4().hex
        merged.append(Interval(cur.start_idx, cur.end_idx, id=preserved_id, source=cur.source))
        return merged

    def union(self) -> List[Interval]:
        allv = [Interval(iv.start_idx, iv.end_idx, id=iv.id, source=iv.source) for iv in (self.manual + self.auto)]
        return self._merge_intervals(allv)

    def set_manual(self, intervals: List[Interval]) -> None:
        self._push_undo()
        self.manual = [iv.normalized() for iv in intervals]

    def add_manual(self, start_idx: int, end_idx: int) -> Interval:
        self._push_undo()
        iv = Interval(int(start_idx), int(end_idx), source="manual")
        self.manual.append(iv)
        self.manual = self._merge_intervals(self.manual)
        return iv

    def remove_by_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        self._push_undo()
        s = set(ids)
        self.manual = [iv for iv in self.manual if iv.id not in s]
        # auto intervals are not removed via selection

    def resize_manual(self, target_id: str, new_start: int, new_end: int) -> None:
        self._push_undo()
        for i, iv in enumerate(self.manual):
            if iv.id == target_id:
                self.manual[i] = Interval(int(new_start), int(new_end), id=iv.id, source="manual").normalized()
                break
        self.manual = self._merge_intervals(self.manual)

    # ---------- Auto computation ----------
    def recompute_auto_edges(self, *, lidar_indices: np.ndarray, lidar_times: np.ndarray, gps_t_min: float, gps_t_max: float, offset_ms: float) -> None:
        """Compute edge ranges where adjusted LiDAR time falls outside GPS range.
        Only creates head/tail intervals; internal gaps are not considered.
        """
        if lidar_indices.size == 0:
            self.auto = []
            return
        adj = lidar_times - float(offset_ms) * 1e-3
        n = adj.size
        # left edge: first index with adj >= gps_t_min
        left = 0
        while left < n and adj[left] < gps_t_min:
            left += 1
        # right edge: last index with adj <= gps_t_max
        right = n - 1
        while right >= 0 and adj[right] > gps_t_max:
            right -= 1
        out: List[Interval] = []
        if left > 0:
            out.append(Interval(int(lidar_indices[0]), int(lidar_indices[left - 1]), source="auto"))
        if right < n - 1:
            out.append(Interval(int(lidar_indices[right + 1]), int(lidar_indices[-1]), source="auto"))
        self.auto = self._merge_intervals(out)

    # ---------- JSON I/O ----------
    def save(self) -> Optional[str]:
        if not (self.data_root and self.worker):
            return None
        if not self.fixed_timestamp:
            self.fixed_timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self._build_new_filename(self.data_root, self.fixed_timestamp, self.worker)
        self.filename = path
        items: List[Dict[str, object]] = []
        union = self.union()
        # deterministic numbering
        for i, iv in enumerate(union, start=1):
            items.append({
                "label": f"start{i}",
                "lidar_idx": int(iv.start_idx),
                "files": {"lidar": self._lidar_path_for_index(iv.start_idx)}
            })
            items.append({
                "label": f"end{i}",
                "lidar_idx": int(iv.end_idx),
                "files": {"lidar": self._lidar_path_for_index(iv.end_idx)}
            })
        try:
            with open(path, "w") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            return path
        except Exception:
            return None

    def _lidar_path_for_index(self, idx: int) -> str:
        # Best-effort path: joins data_root with lidar_xyzi and a wildcard-like file.
        # Since exact filename may include timestamps, we cannot reconstruct perfectly without scans list.
        # We record a canonical pattern path for user reference.
        root = self.data_root or "."
        return os.path.join(root, "lidar_xyzi", f"lidar_{int(idx):06d}_*.bin")

    def load_from_json(self, json_path: str) -> None:
        with open(json_path, "r") as f:
            arr = json.load(f)
        # determine root from path
        root = os.path.abspath(os.path.join(os.path.dirname(json_path), os.pardir))
        if os.path.basename(os.path.dirname(json_path)) == "marks_json":
            root = os.path.abspath(os.path.join(os.path.dirname(json_path), os.pardir))
        self.data_root = root
        name = os.path.basename(json_path)
        # parse fixed timestamp and worker if possible: gnss_{folder}_{ts}_{worker}.json
        try:
            base = os.path.basename(root)
            prefix = f"gnss_{base}_"
            rest = name[len(prefix):]
            ts, worker = rest.rsplit("_", 1)
            worker = worker[:-5] if worker.lower().endswith(".json") else worker
            self.fixed_timestamp = ts
            self.worker = worker
        except Exception:
            pass
        # parse intervals
        items: List[Interval] = []
        try:
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                lbl = str(obj.get("label", ""))
                idx = int(obj.get("lidar_idx"))
                if lbl.startswith("start"):
                    items.append(Interval(idx, idx))  # temp pair start, will pair below
                elif lbl.startswith("end") and items:
                    last = items[-1]
                    items[-1] = Interval(last.start_idx, idx)
        except Exception:
            items = []
        self.manual = self._merge_intervals([Interval(iv.start_idx, iv.end_idx, source="manual") for iv in items])
        self.auto = []
        self.filename = json_path

    # ---------- Utilities ----------
    @staticmethod
    def list_candidate_jsons(root: str) -> List[str]:
        d = os.path.join(root, "marks_json")
        if not os.path.isdir(d):
            return []
        base = os.path.basename(os.path.normpath(root))
        pref = f"gnss_{base}_"
        out = [os.path.join(d, f) for f in os.listdir(d) if f.startswith(pref) and f.endswith('.json')]
        out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return out