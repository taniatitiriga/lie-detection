from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


# ── column definitions ────────────────────────────────────────────────────────

AU_C_COLUMNS: List[str] = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

AU_R_COLUMNS: List[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

GAZE_COLUMNS: List[str] = ["gaze_angle_x", "gaze_angle_y"]

POSE_COLUMNS: List[str] = ["pose_Rx", "pose_Ry", "pose_Rz"]

ALL_SIGNAL_COLUMNS: List[str] = AU_C_COLUMNS + AU_R_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS

# Threshold: columns with >90% zero frames across all clips are pruned.
DEAD_COL_ZERO_FRAC: float = 0.90

N_WINDOWS: int = 3
WINDOW_STAT_SUFFIXES: List[str] = ["mean", "std", "madiff"]
DELTA_STAT_SUFFIXES: List[str] = ["dmean", "dstd"]


# ── stat helpers ──────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return math.sqrt(sum((x - mu) ** 2 for x in values) / n)


def _madiff(values: List[float]) -> float:
    """Mean absolute frame-to-frame difference."""
    n = len(values)
    if n < 2:
        return 0.0
    return sum(abs(values[i] - values[i - 1]) for i in range(1, n)) / (n - 1)


def _window_stats(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, std, madiff) for one window."""
    return (_mean(values), _std(values), _madiff(values))


def _delta_stats(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) of frame-to-frame differences for one window."""
    if len(values) < 2:
        return (0.0, 0.0)
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    return (_mean(deltas), _std(deltas))


def _split_into_windows(values: List[float], n_windows: int = N_WINDOWS) -> List[List[float]]:
    """Split a frame-level list into n roughly-equal temporal windows."""
    n = len(values)
    if n == 0:
        return [[] for _ in range(n_windows)]
    base_size = n // n_windows
    remainder = n % n_windows
    windows: List[List[float]] = []
    start = 0
    for w in range(n_windows):
        end = start + base_size + (1 if w < remainder else 0)
        windows.append(values[start:end])
        start = end
    return windows


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


# ── dead-column detection (pre-scan) ─────────────────────────────────────────

def _find_dead_columns(
    clip_csv_paths: List[Path],
    confidence_threshold: float,
    zero_frac_threshold: float = DEAD_COL_ZERO_FRAC,
) -> set[str]:
    """
    Pre-scan all clip CSVs and return columns where >zero_frac_threshold of
    all high-confidence frames are zero.  Gaze and pose columns are never pruned.
    """
    prunable = set(AU_C_COLUMNS + AU_R_COLUMNS)
    total_frames: Dict[str, int] = {col: 0 for col in prunable}
    zero_frames: Dict[str, int] = {col: 0 for col in prunable}

    for path in clip_csv_paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            available = set(reader.fieldnames)
            cols_here = [c for c in prunable if c in available]

            for row in reader:
                if float(row.get("confidence", "0")) < confidence_threshold:
                    continue
                for col in cols_here:
                    total_frames[col] += 1
                    if float(row[col]) == 0.0:
                        zero_frames[col] += 1

    dead: set[str] = set()
    for col in prunable:
        if total_frames[col] == 0:
            dead.add(col)
        elif zero_frames[col] / total_frames[col] > zero_frac_threshold:
            dead.add(col)

    return dead


# ── per-clip feature extractor (windowed + deltas) ───────────────────────────

def _read_clip_frames(
    clip_csv_path: Path,
    confidence_threshold: float,
    live_columns: List[str],
    present_au_r: List[str],
) -> Dict[str, List[float]]:
    """Read one clip CSV and return per-column frame-level lists."""
    acc: Dict[str, List[float]] = {col: [] for col in live_columns}

    with clip_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty OpenFace CSV: {clip_csv_path}")

        for row in reader:
            confidence = float(row.get("confidence", "0"))
            if confidence < confidence_threshold:
                continue

            for col in live_columns:
                if col in present_au_r or col not in AU_R_COLUMNS:
                    acc[col].append(float(row.get(col, "0")))
                else:
                    acc[col].append(0.0)

    return acc


def _clip_features_windowed(
    acc: Dict[str, List[float]],
    live_columns: List[str],
    live_au_r: List[str],
) -> Tuple[Dict[str, float], bool]:
    """
    Compute windowed statistics and delta features from frame-level data.

    For each live column: 3 windows x 3 stats (mean, std, madiff).
    For each live AU_r column: 3 windows x 2 delta stats (dmean, dstd).
    """
    has_frames = any(len(v) > 0 for v in acc.values())
    features: Dict[str, float] = {}

    for col in live_columns:
        windows = _split_into_windows(acc[col])
        for w_idx, window in enumerate(windows):
            stats = _window_stats(window)
            for suffix, val in zip(WINDOW_STAT_SUFFIXES, stats):
                features[f"{col}_w{w_idx}_{suffix}"] = val

    for col in live_au_r:
        windows = _split_into_windows(acc[col])
        for w_idx, window in enumerate(windows):
            dm, ds = _delta_stats(window)
            features[f"{col}_w{w_idx}_dmean"] = dm
            features[f"{col}_w{w_idx}_dstd"] = ds

    return features, has_frames


# ── header builder ────────────────────────────────────────────────────────────

def _build_header(live_columns: List[str], live_au_r: List[str]) -> List[str]:
    """Construct the full ordered header (excluding meta columns)."""
    cols: List[str] = []
    for col in live_columns:
        for w_idx in range(N_WINDOWS):
            for suffix in WINDOW_STAT_SUFFIXES:
                cols.append(f"{col}_w{w_idx}_{suffix}")
    for col in live_au_r:
        for w_idx in range(N_WINDOWS):
            for suffix in DELTA_STAT_SUFFIXES:
                cols.append(f"{col}_w{w_idx}_{suffix}")
    return cols


# ── main builder ──────────────────────────────────────────────────────────────

def build_visual_features_csv(
    labels_csv_path: Path = Path("data/labels.csv"),
    extracted_au_dir: Path = Path("data/extracted_AU_gaze"),
    output_csv_path: Path = Path("features/visual.csv"),
    confidence_threshold: float = 0.9,
) -> None:
    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _read_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-scan: identify dead columns across dataset.
    clip_csv_paths = [
        extracted_au_dir / f"{Path(cid).stem}.csv" for cid in clip_ids
    ]
    dead_cols = _find_dead_columns(clip_csv_paths, confidence_threshold)
    if dead_cols:
        print(f"Pruned {len(dead_cols)} dead columns: {sorted(dead_cols)}")

    live_columns = [c for c in ALL_SIGNAL_COLUMNS if c not in dead_cols]
    live_au_r = [c for c in AU_R_COLUMNS if c not in dead_cols]

    feature_cols = _build_header(live_columns, live_au_r)
    header = ["clip_id", "subject_id", "is_deceptive"] + feature_cols

    # Determine which AU_r are actually present per CSV (done once with first existing CSV).
    present_au_r_set: set[str] | None = None

    rows: List[List[object]] = []
    total = len(clip_ids)

    for i, clip_id in enumerate(clip_ids, start=1):
        stem = Path(clip_id).stem
        clip_csv_path = extracted_au_dir / f"{stem}.csv"

        if present_au_r_set is None:
            with clip_csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                available = set(reader.fieldnames or [])
                present_au_r_set = {c for c in live_au_r if c in available}

        present_au_r_list = [c for c in live_au_r if c in present_au_r_set]

        acc = _read_clip_frames(
            clip_csv_path, confidence_threshold, live_columns, present_au_r_list,
        )

        features, has_high_conf_frames = _clip_features_windowed(
            acc, live_columns, live_au_r,
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
    n_windowed = len(live_columns) * N_WINDOWS * len(WINDOW_STAT_SUFFIXES)
    n_delta = len(live_au_r) * N_WINDOWS * len(DELTA_STAT_SUFFIXES)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"Done — {output_csv_path.as_posix()} written "
        f"({len(rows)} rows, {n_features} features: "
        f"{n_windowed} windowed + {n_delta} delta, "
        f"{len(dead_cols)} columns pruned)"
    )


if __name__ == "__main__":
    build_visual_features_csv()
