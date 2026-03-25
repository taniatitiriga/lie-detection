from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


# These are the 18 binary AU "c" columns we want to aggregate.
AU_COLUMNS = [
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


def run_openface_feature_extraction(
    input_dir: Path = Path("/data/raw_videos"),
    output_dir: Path = Path("/data/extracted_AU_gaze"),
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

    cmd = [str(feature_extraction_bin)]  # ignore this RIP
    for f in files:
        cmd.extend(["-f", str(f)])

    cmd.extend(["-out_dir", str(output_dir)])

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


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


def _clip_au_means_from_csv(
    clip_csv_path: Path,
    confidence_threshold: float,
) -> Tuple[List[float], bool]:
    """Return (18 AU means, whether any high-confidence frames were present)."""
    au_sums = {col: 0.0 for col in AU_COLUMNS}
    high_conf_count = 0

    with clip_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty OpenFace CSV: {clip_csv_path}")

        missing = [c for c in AU_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing AU columns in {clip_csv_path.name}: {missing}")
        if "confidence" not in reader.fieldnames:
            raise KeyError(f"Missing `confidence` column in {clip_csv_path.name}")

        for row in reader:
            confidence = float(row["confidence"])
            if confidence < confidence_threshold:
                continue

            high_conf_count += 1
            for col in AU_COLUMNS:
                au_sums[col] += float(row[col])

    if high_conf_count == 0:
        return [0.0] * len(AU_COLUMNS), False

    return [au_sums[col] / high_conf_count for col in AU_COLUMNS], True


def build_visual_features_csv(
    labels_csv_path: Path = Path("data/labels.csv"),
    extracted_au_dir: Path = Path("features/extracted_AU_gaze"),
    output_csv_path: Path = Path("features/visual.csv"),
    confidence_threshold: float = 0.9,
) -> None:
    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _read_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[List[object]] = []
    total = len(clip_ids)

    for i, clip_id in enumerate(clip_ids, start=1):
        stem = Path(clip_id).stem  # strip .mp4 -> trial_lie_001
        clip_csv_path = extracted_au_dir / f"{stem}.csv"

        au_values, has_high_conf_frames = _clip_au_means_from_csv(
            clip_csv_path=clip_csv_path,
            confidence_threshold=confidence_threshold,
        )

        if not has_high_conf_frames:
            print(f"WARNING: {clip_id} has no high-confidence frames")

        rows.append(
            [
                clip_id,
                subject_id_by_clip[clip_id],
                is_deceptive_by_clip[clip_id],
                *au_values,
            ]
        )

        if i % 10 == 0:
            print(f"Progress: {i}/{total} clips")

    header = ["clip_id", "subject_id", "is_deceptive", *AU_COLUMNS]
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"Done — {output_csv_path.as_posix()} written ({len(rows)} rows, 18 AU features)"
    )


if __name__ == "__main__":
    build_visual_features_csv()
