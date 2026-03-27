from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


# ── column definitions ────────────────────────────────────────────────────────

# 18 binary AU columns (presence/absence)
AU_C_COLUMNS: List[str] = [
    "AU01_c",
    "AU02_c",
    "AU04_c",
    "AU05_c",
    "AU06_c",
    "AU07_c",
    "AU09_c",
    "AU10_c",
    "AU12_c",
    "AU14_c",
    "AU15_c",
    "AU17_c",
    "AU20_c",
    "AU23_c",
    "AU25_c",
    "AU26_c",
    "AU28_c",
    "AU45_c",
]

# 17 AU intensity (regression) columns – AU28 has no _r counterpart in OpenFace
AU_R_COLUMNS: List[str] = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
]

GAZE_COLUMNS: List[str] = ["gaze_angle_x", "gaze_angle_y"]

POSE_COLUMNS: List[str] = ["pose_Rx", "pose_Ry", "pose_Rz"]


# ── 6-stat temporal descriptor (matches experiment_RF.py) ─────────────────────

# Stat suffixes in fixed order — every signal column gets all 6.
STAT_SUFFIXES: List[str] = ["mean", "std", "med", "iqr", "mad", "madiff"]


def _six_stats(values: List[float]) -> Tuple[float, ...]:
    """
    Compute 6 descriptors from a time-series of frame-level values:
       mean, std, median, IQR, MAD, mean-abs-frame-diff

    Returns a tuple of 6 floats (all 0.0 if *values* is empty).
    """
    n = len(values)
    if n == 0:
        return (0.0,) * 6

    arr = sorted(values)  # needed for percentiles / median

    mu = sum(values) / n
    if n == 1:
        return (mu, 0.0, mu, 0.0, 0.0, 0.0)

    variance = sum((x - mu) ** 2 for x in values) / n
    std = math.sqrt(variance)

    # median
    mid = n // 2
    med = (arr[mid] + arr[mid - 1]) / 2.0 if n % 2 == 0 else arr[mid]

    # IQR  (Q75 – Q25, linear interpolation)
    def _pct(sorted_vals: List[float], p: float) -> float:
        k = (len(sorted_vals) - 1) * p
        lo = int(math.floor(k))
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = k - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    q25 = _pct(arr, 0.25)
    q75 = _pct(arr, 0.75)
    iqr = q75 - q25

    # MAD (median absolute deviation)
    abs_devs = sorted(abs(v - med) for v in values)
    mad_mid = len(abs_devs) // 2
    mad = (
        (abs_devs[mad_mid] + abs_devs[mad_mid - 1]) / 2.0
        if len(abs_devs) % 2 == 0
        else abs_devs[mad_mid]
    )

    # Mean absolute frame-to-frame difference
    madiff = sum(abs(values[i] - values[i - 1]) for i in range(1, n)) / (n - 1)

    return (mu, std, med, iqr, mad, madiff)


# ── OpenFace runner (unchanged) ───────────────────────────────────────────────

def run_openface_feature_extraction(
    input_dir: Path = Path("data/raw_videos"),
    output_dir: Path = Path("data/extracted_AU_gaze"),
    feature_extraction_bin: Path = Path(
        "/mnt/c/users/tania/tools/openface/build/bin/FeatureExtraction"
    ),
) -> None:
    """Run OpenFace FeatureExtraction (only needed to generate frame-level CSVs)."""
    files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".mp4"}
    )
    if not files:
        print("No input files found")
        return

    cmd = [str(feature_extraction_bin)]
    for f in files:
        cmd.extend(["-f", str(f)])

    cmd.extend(["-out_dir", str(output_dir)])

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


# ── label reader (unchanged) ──────────────────────────────────────────────────

def _read_labels(
    labels_csv_path: Path,
) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    """Return clip_ids in file order + subject_id and is_deceptive mappings."""
    clip_ids: List[str] = []
    subject_id_by_clip: Dict[str, str] = {}
    is_deceptive_by_clip: Dict[str, int] = {}

    with labels_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "clip_id" not in fieldnames or "subject_id" not in fieldnames:
            raise KeyError("Expected `clip_id` and `subject_id` columns in labels.csv")

        has_is_deceptive = "is_deceptive" in fieldnames
        has_label = "label" in fieldnames
        if not has_is_deceptive and not has_label:
            raise KeyError("Expected `is_deceptive` (or fallback `label`) column in labels.csv")

        for row in reader:
            clip_id = row["clip_id"]
            clip_ids.append(clip_id)
            subject_id_by_clip[clip_id] = row["subject_id"]

            if has_is_deceptive:
                is_deceptive_by_clip[clip_id] = int(row["is_deceptive"])
            else:
                label = str(row["label"]).strip().lower()
                is_deceptive_by_clip[clip_id] = 1 if label == "deceptive" else 0

    return clip_ids, subject_id_by_clip, is_deceptive_by_clip


# ── per-clip feature extractor ────────────────────────────────────────────────

def _clip_features_from_csv(
    clip_csv_path: Path,
    confidence_threshold: float,
) -> Tuple[Dict[str, float], bool]:
    """
    Aggregate frame-level OpenFace features into a single per-clip feature dict.

    Returns
    -------
    features : dict  {column_name -> value}
    has_high_conf_frames : bool
    """
    # accumulators: list of per-frame values for each column of interest
    all_cols = AU_C_COLUMNS + AU_R_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS

    acc: Dict[str, List[float]] = {col: [] for col in all_cols}

    with clip_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty OpenFace CSV: {clip_csv_path}")

        available = set(reader.fieldnames)

        # Validate required columns
        all_required = AU_C_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS
        missing = [c for c in all_required if c not in available]
        if missing:
            raise KeyError(f"Missing columns in {clip_csv_path.name}: {missing}")

        if "confidence" not in available:
            raise KeyError(f"Missing `confidence` column in {clip_csv_path.name}")

        # AU_R columns may legitimately be absent for some AU numbers
        present_au_r = [c for c in AU_R_COLUMNS if c in available]
        absent_au_r = [c for c in AU_R_COLUMNS if c not in available]
        if absent_au_r:
            print(f"  INFO: {clip_csv_path.name} is missing AU_r columns: {absent_au_r}")

        for row in reader:
            confidence = float(row["confidence"])
            if confidence < confidence_threshold:
                continue

            for col in all_required:
                acc[col].append(float(row[col]))
            for col in present_au_r:
                acc[col].append(float(row[col]))

    has_frames = any(len(v) > 0 for v in acc.values())

    features: Dict[str, float] = {}

    # All signal groups get the same 6 temporal descriptors.
    for col in AU_C_COLUMNS + AU_R_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS:
        stats = _six_stats(acc[col])
        for suffix, val in zip(STAT_SUFFIXES, stats):
            features[f"{col}_{suffix}"] = val

    return features, has_frames


# ── main builder ──────────────────────────────────────────────────────────────

def _build_header() -> List[str]:
    """Construct the full ordered header list (excluding clip_id / subject_id / is_deceptive)."""
    cols: List[str] = []
    for col in AU_C_COLUMNS + AU_R_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS:
        for suffix in STAT_SUFFIXES:
            cols.append(f"{col}_{suffix}")
    return cols


def build_visual_features_csv(
    labels_csv_path: Path = Path("data/labels.csv"),
    extracted_au_dir: Path = Path("data/extracted_AU_gaze"),
    output_csv_path: Path = Path("features/visual.csv"),
    confidence_threshold: float = 0.9,
) -> None:
    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _read_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = _build_header()
    header = ["clip_id", "subject_id", "is_deceptive"] + feature_cols

    rows: List[List[object]] = []
    total = len(clip_ids)

    for i, clip_id in enumerate(clip_ids, start=1):
        stem = Path(clip_id).stem          # strip .mp4  ->  trial_lie_001
        clip_csv_path = extracted_au_dir / f"{stem}.csv"

        features, has_high_conf_frames = _clip_features_from_csv(
            clip_csv_path=clip_csv_path,
            confidence_threshold=confidence_threshold,
        )

        if not has_high_conf_frames:
            print(f"WARNING: {clip_id} has no high-confidence frames")

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
        ]
        for col in feature_cols:
            row.append(features.get(col, 0.0))

        rows.append(row)

        if i % 10 == 0:
            print(f"Progress: {i}/{total} clips")

    n_features = len(feature_cols)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"Done — {output_csv_path.as_posix()} written "
        f"({len(rows)} rows, {n_features} visual features)"
    )


if __name__ == "__main__":
    build_visual_features_csv()
