from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from typing import Dict, List


def load_features(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load each CSV in paths, align by clip_id (inner join on clip_id column).
    Assert all CSVs have identical clip_id sets.
    Drop clip_id, subject_id, is_deceptive from feature matrix.

    Returns:
      X: (n_clips, n_features)
      y: (n_clips,)
      subject_ids: list (n_clips)
      clip_ids: list (n_clips)
    """
    if len(paths) == 0:
        raise ValueError("Expected at least one CSV path in `paths`")

    meta_cols = {"clip_id", "subject_id", "is_deceptive"}

    merged_features_by_clip: Dict[str, List[float]] = {}
    subject_id_by_clip: Dict[str, str] = {}
    y_by_clip: Dict[str, int] = {}
    clip_id_order: List[str] = []

    feature_cols_seen: set[str] = set()
    expected_clip_set: set[str] | None = None

    for csv_idx, p in enumerate(paths):
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {p}")

            missing = meta_cols - set(reader.fieldnames)
            if missing:
                raise KeyError(f"File {p} is missing required columns: {sorted(missing)}")

            feature_cols = [c for c in reader.fieldnames if c not in meta_cols]

            overlap = feature_cols_seen.intersection(feature_cols)
            if overlap:
                raise ValueError(f"Overlapping feature column names across CSVs: {sorted(overlap)}")
            feature_cols_seen |= set(feature_cols)

            clip_set_this: set[str] = set()

            for row in reader:
                cid = row["clip_id"]
                clip_set_this.add(cid)

                if csv_idx == 0:
                    clip_id_order.append(cid)
                    subject_id_by_clip[cid] = row["subject_id"]
                    y_by_clip[cid] = int(row["is_deceptive"])
                    merged_features_by_clip[cid] = []

                if cid not in merged_features_by_clip:
                    raise AssertionError(f"clip_id {cid} present in CSV[{csv_idx}] but not in CSV[0]")

                vals: List[float] = []
                for c in feature_cols:
                    raw = row.get(c, "")
                    vals.append(float(raw) if raw != "" else 0.0)
                merged_features_by_clip[cid].extend(vals)

        if expected_clip_set is None:
            expected_clip_set = clip_set_this
        else:
            if clip_set_this != expected_clip_set:
                raise AssertionError(
                    "clip_id sets differ between CSVs "
                    f"(expected {len(expected_clip_set)}, got {len(clip_set_this)})"
                )

    if expected_clip_set is None:
        raise ValueError("No CSVs loaded")

    if len(clip_id_order) != len(expected_clip_set):
        raise AssertionError("clip_id cardinality mismatch")

    first_cid = clip_id_order[0]
    n_clips = len(clip_id_order)
    n_features = len(merged_features_by_clip[first_cid])

    X = np.zeros((n_clips, n_features), dtype=float)
    y = np.zeros((n_clips,), dtype=int)
    subject_ids: List[str] = []

    for i, cid in enumerate(clip_id_order):
        X[i, :] = np.asarray(merged_features_by_clip[cid], dtype=float)
        y[i] = y_by_clip[cid]
        subject_ids.append(subject_id_by_clip[cid])

    return X, y, subject_ids, clip_id_order


def load_single_modality(path: str) -> np.ndarray:
    """Load a single CSV and return only the feature columns as ndarray."""
    meta_cols = {"clip_id", "subject_id", "is_deceptive"}
    rows: list[list[float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        feature_cols = [c for c in reader.fieldnames if c not in meta_cols]
        for row in reader:
            vals = [float(row.get(c, "") or 0.0) for c in feature_cols]
            rows.append(vals)
    return np.array(rows, dtype=float)



