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
        # data_root now represents the directory where marks JSONs are stored
        self.data_root: Optional[str] = data_root
        self.worker: str = worker
        self.fixed_timestamp: Optional[str] = None  # YYYYMMDD_HHMMSS kept constant for this session file
        self.filename: Optional[str] = None
        self.manual: List[Interval] = []
        self.auto: List[Interval] = []
        self.base_name: Optional[str] = None  # used in filename pattern gnss_{base}_...
        # undo/redo history as list of (manual_copy, auto_copy)
        self._undo: List[Tuple[List[Interval], List[Interval]]] = []
        self._redo: List[Tuple[List[Interval], List[Interval]]] = []

    # ---------- Persistence ----------
    def _ensure_marks_dir(self, root: str) -> str:
        # treat root as the marks directory directly
        os.makedirs(root, exist_ok=True)
        return root

    def set_base_name(self, base: str) -> None:
        self.base_name = (base or "").strip() or None

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
        marks_dir = self._ensure_marks_dir(root)
        base = (self.base_name or os.path.basename(os.path.normpath(root)))
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
    def recompute_auto_edges(self, *, lidar_indices: np.ndarray, lidar_times: np.ndarray, gps_t_min: float, gps_t_max: float, offset_ms: float, mark_out_of_range: bool = False) -> None:
        if lidar_indices.size == 0:
            self.auto = []
            return
        # 외삽 허용 기본값: 범위 밖 구간을 '무효'로 표시하지 않음
        if not mark_out_of_range:
            self.auto = []
            return

        adj = lidar_times - float(offset_ms) * 1e-3
        n = adj.size
        left = 0
        while left < n and adj[left] < gps_t_min:
            left += 1
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
        for i, iv in enumerate(union, start=1):
            items.append({
                "label": f"start{i}",
                "lidar_idx": int(iv.start_idx)
            })
            items.append({
                "label": f"end{i}",
                "lidar_idx": int(iv.end_idx)
            })
        try:
            with open(path, "w") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            return path
        except Exception:
            return None

    def load_from_json(self, json_path: str) -> None:
        with open(json_path, "r") as f:
            arr = json.load(f)
        # marks directory is parent path now
        root = os.path.abspath(os.path.dirname(json_path))
        self.data_root = root
        name = os.path.basename(json_path)
        try:
            # parse schema gnss_{base}_{ts}_{worker}.json
            rest = name
            if rest.startswith("gnss_"):
                rest = rest[len("gnss_"):]
            parts = rest.split("_")
            if len(parts) >= 3:
                self.base_name = parts[0]
                self.fixed_timestamp = parts[1]
                worker_part = "_".join(parts[2:])
                if worker_part.lower().endswith(".json"):
                    worker_part = worker_part[:-5]
                self.worker = worker_part
        except Exception:
            pass
        items: List[Interval] = []
        try:
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                lbl = str(obj.get("label", ""))
                idx = int(obj.get("lidar_idx"))
                if lbl.startswith("start"):
                    items.append(Interval(idx, idx))
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
        if not os.path.isdir(root):
            return []
        out = [os.path.join(root, f) for f in os.listdir(root) if f.startswith("gnss_") and f.endswith('.json')]
        out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return out