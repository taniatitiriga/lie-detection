#!/usr/bin/env python3

"""
Random Forest prediction on facial AU's and eye gaze from OpenFace CSVs.

Usage (from repo root):
    uv run playground/experiment_rf.py --data-dir data/res --out runs/rf_exp1

It:

- parses labels from filenames,
- extracts robust aggregate features (AUs, gaze, pose, landmarks — mean/std/median/IQR/mean-abs-diff, AU activation counts and blink-rate if AU45 available),
- builds X, y,
- runs Stratified K-Fold CV (default 5 folds),
- reports accuracy/AUC/F1/precision/recall + confusion matrix,
- writes features.csv, cv_results.json, and saves the final fitted RF model.
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List


def parse_metadata(path: Path) -> Tuple[int, int]:
    """
    Parse filename like: trial_lie_03.csv -> (label_int, trial_number)
    label_int: 1 for 'lie', 0 for 'truth'
    """
    name = path.stem
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path.name}")
    _, label_str, num = parts[:3]
    label = 1 if label_str.lower().startswith("lie") else 0
    try:
        num_i = int(num)
    except ValueError:
        # attempt to strip leading zeros or non-digit suffixes
        num_i = int(''.join(ch for ch in num if ch.isdigit()) or 0)
    return label, num_i


def available_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def au_r_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.upper().startswith("AU") and c.endswith("_r")]


def au_c_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.upper().startswith("AU") and c.endswith("_c")]


def landmark_xy_cols(df: pd.DataFrame) -> List[str]:
    # x_0 ... x_67, y_0 ... y_67
    xs = [f"x_{i}" for i in range(68) if f"x_{i}" in df.columns]
    ys = [f"y_{i}" for i in range(68) if f"y_{i}" in df.columns]
    return xs + ys


def gaze_cols(df: pd.DataFrame) -> List[str]:
    possible = [
        "gaze_angle_x", "gaze_angle_y",
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
    ]
    return [c for c in possible if c in df.columns]


def pose_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("pose_")]


def compute_basic_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute (mean, std, median, iqr, mad, mean_abs_diff) across time for each column and flatten."""
    out = {}
    if arr.size == 0:
        return out
    # if arr is 1D, make it 2D
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # compute per-column stats
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    medians = np.nanmedian(arr, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    iqr = q75 - q25
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr, axis=0)), axis=0)
    # mean absolute frame-to-frame difference
    diffs = np.nanmean(np.abs(np.diff(arr, axis=0)), axis=0) if arr.shape[0] >= 2 else np.zeros(arr.shape[1])
    # pack into dict
    for i, (m, s, med, ii, mm, d) in enumerate(zip(means, stds, medians, iqr, mad, diffs)):
        out[f"m_{i}"] = float(m)
        out[f"s_{i}"] = float(s)
        out[f"med_{i}"] = float(med)
        out[f"iqr_{i}"] = float(ii)
        out[f"mad_{i}"] = float(mm)
        out[f"madiff_{i}"] = float(d)
    return out


def aggregate_one(csv_path: Path, conf_thr: float = 0.0) -> Dict[str, Any]:
    """
    Read CSV and produce an aggregated feature dict for the video.
    """
    df = pd.read_csv(csv_path)
    # filter low-confidence frames (if confidence exists)
    if "confidence" in df.columns:
        df = df[df["confidence"].astype(float) >= conf_thr]
    feats: Dict[str, Any] = {}
    feats["frames_kept"] = int(len(df))
    feats["conf_mean"] = float(df["confidence"].mean()) if "confidence" in df.columns and len(df) > 0 else 0.0

    # AUs (regression)
    au_r = au_r_cols(df)
    if au_r:
        arr = df[au_r].to_numpy(dtype=float)
        stats = compute_basic_stats(arr)
        # prefix keys so they don't collide
        feats.update({f"AU_r_{k}": v for k, v in stats.items()})

    # AUs (classification) - counts and mean (blink-like rates)
    au_c = au_c_cols(df)
    for c in au_c:
        feats[f"{c}_count"] = int((df[c] == 1).sum())
        feats[f"{c}_mean"] = float(df[c].mean())

    # Blink rate if AU45_c available
    if "AU45_c" in df.columns:
        feats["blink_rate"] = float((df["AU45_c"] == 1).sum()) / max(1, len(df))

    # gaze
    gcols = gaze_cols(df)
    if gcols:
        arr = df[gcols].to_numpy(dtype=float)
        stats = compute_basic_stats(arr)
        feats.update({f"g_{k}": v for k, v in stats.items()})

    # pose
    pcols = pose_cols(df)
    if pcols:
        arr = df[pcols].to_numpy(dtype=float)
        stats = compute_basic_stats(arr)
        feats.update({f"pose_{k}": v for k, v in stats.items()})

    # landmarks (x,y)
    lmcols = landmark_xy_cols(df)
    if lmcols:
        arr = df[lmcols].to_numpy(dtype=float)
        # we don't want 136*6 features explosion; reduce with PCA-like stats by grouping pairs
        # but for simplicity we compute the same basic stats per landmark coordinate
        stats = compute_basic_stats(arr)
        feats.update({f"lm_{k}": v for k, v in stats.items()})

    return feats


def build_dataset(data_dir: Path, conf_thr: float = 0.0) -> Tuple[pd.DataFrame, np.ndarray]:
    rows = []
    labels = []
    csvs = sorted(data_dir.glob("trial_*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir} (pattern trial_*.csv)")
    for csv in csvs:
        label, trial_id = parse_metadata(csv)
        feats = aggregate_one(csv, conf_thr=conf_thr)
        feats["file"] = str(csv.name)
        feats["trial_id"] = trial_id
        rows.append(feats)
        labels.append(label)
    Xdf = pd.DataFrame(rows).fillna(0)
    y = np.array(labels, dtype=int)
    return Xdf, y


def run_cv_and_train(Xdf: pd.DataFrame, y: np.ndarray, out_dir: Path, n_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # features: drop metadata columns
    meta_cols = {"file", "trial_id"}
    feature_cols = [c for c in Xdf.columns if c not in meta_cols]
    X = Xdf[feature_cols].to_numpy(dtype=float)
    # pipeline: scaler (useful if you switch to LR/SVM) + RF
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # cross_val_predict to get per-sample predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict", n_jobs=-1)
    # probabilities for AUC
    try:
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    except Exception:
        y_proba = None

    # metrics
    results = {}
    results["accuracy"] = float(accuracy_score(y, y_pred))
    results["f1"] = float(f1_score(y, y_pred))
    results["precision"] = float(precision_score(y, y_pred))
    results["recall"] = float(recall_score(y, y_pred))
    if y_proba is not None and len(np.unique(y)) == 2:
        try:
            results["roc_auc"] = float(roc_auc_score(y, y_proba))
        except Exception:
            results["roc_auc"] = None
    else:
        results["roc_auc"] = None
    results["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()

    # Fit final model on full data
    pipeline.fit(X, y)
    model_path = out_dir / "rf_model.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_cols}, model_path)

    # save features & preds
    Xdf_out = Xdf.copy()
    Xdf_out["y"] = y
    Xdf_out["y_pred"] = y_pred
    if y_proba is not None:
        Xdf_out["y_proba"] = y_proba
    Xdf_out.to_csv(out_dir / "features_with_preds.csv", index=False)

    # save results
    (out_dir / "cv_results.json").write_text(json.dumps(results, indent=2))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/extracted_AU_gaze", help="Path to folder with trial_*.csv")
    p.add_argument("--out", type=str, default="runs/experiment_rf_001", help="Output folder to save model/results")
    p.add_argument("--conf-thr", type=float, default=0.0, help="Confidence threshold to filter frames (0-1)")
    p.add_argument("--n-splits", type=int, default=5, help="Stratified K folds")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    out_dir = (root / args.out).resolve()

    print(f"DATA DIR: {data_dir}")
    print(f"OUTPUT: {out_dir}")
    print("Building dataset (this may take a moment)...")
    Xdf, y = build_dataset(data_dir, conf_thr=args.conf_thr)
    print(f"Found {len(Xdf)} trials. Feature vector size: {Xdf.shape[1]} (including metadata)")

    print("Running CV + training RandomForest...")
    results = run_cv_and_train(Xdf, y, out_dir, n_splits=args.n_splits, random_state=args.random_state)

    print("Results:")
    print(json.dumps(results, indent=2))
    print(f"Saved model+features+results to {out_dir}")

if __name__ == "__main__":
    main()
