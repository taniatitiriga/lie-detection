#!/usr/bin/env python3

"""
Random Forest experiment for lie prediction on facial AU's and eye gaze from OpenFace CSVs.

Usage (from repo root):
    uv run playground/experiment_rf.py --data-dir data/res --out runs/rf_exp1

- parses labels from filenames,
- extracts robust aggregate features (AUs, gaze, pose, landmarks — mean/std/median/IQR/mean-abs-diff, AU activation counts and blink-rate if AU45 available),
- builds X, y,
- runs Stratified K-Fold CV (default 5 folds),
- reports accuracy/AUC/F1/precision/recall + confusion matrix,
- writes features.csv, cv_results.json, and saves the final fitted RF model.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import numpy as np
try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

try:
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
except ModuleNotFoundError:
    RandomForestClassifier = None  # type: ignore[assignment]
    StratifiedKFold = None  # type: ignore[assignment]
    cross_val_predict = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]

from typing import Dict, Any, Tuple, List

try:
    import joblib  # type: ignore
except ModuleNotFoundError:
    joblib = None  # type: ignore[assignment]


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
                    raise AssertionError(
                        f"clip_id {cid} present in CSV[{csv_idx}] but not in CSV[0]"
                    )

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


def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list,
    clip_ids: list,
    clf_factory,
    scaler: bool = True,
    n_runs: int = 3,
) -> tuple[float, float, list]:
    """
    Subject-aware LOOCV with clip-level evaluation.

    Returns:
      mean_acc, std_acc, first_run_preds, auc (if computable else nan)
    """
    if StandardScaler is None or roc_auc_score is None:
        raise ModuleNotFoundError("scikit-learn is required for run_loocv()")

    unique_subjects = sorted(set(subject_ids))  # subject-level splits
    subject_ids_arr = np.array(subject_ids)
    clip_ids_arr = np.array(clip_ids)
    y_arr = np.array(y, dtype=int)

    all_preds: list[list[tuple]] = []  # per run

    for run in range(n_runs):
        run_preds = []
        for subj in unique_subjects:
            test_mask = subject_ids_arr == subj
            train_mask = ~test_mask

            # ASSERT: no subject appears in both
            assert not np.any(subject_ids_arr[train_mask] == subj)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y_arr[train_mask], y_arr[test_mask]

            if scaler:
                sc = StandardScaler().fit(X_train)  # fit on train only
                X_train = sc.transform(X_train)
                X_test = sc.transform(X_test)

            clf = clf_factory(run_seed=run)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            test_indices = np.where(test_mask)[0]
            ids = clip_ids_arr[test_indices].tolist()
            for cid, yt, yp, pr in zip(ids, y_test, preds, probs, strict=False):
                run_preds.append((cid, int(yt), int(yp), float(pr)))

        all_preds.append(run_preds)

    # Average accuracy across runs
    run_accs = [np.mean([r[1] == r[2] for r in rp]) for rp in all_preds]
    mean_acc = float(np.mean(run_accs))
    std_acc = float(np.std(run_accs))

    # AUC from averaged probabilities if available.
    auc = float("nan")
    try:
        if len(np.unique(y_arr)) == 2:
            # Aggregate per clip across runs
            prob_sum: Dict[str, float] = {}
            prob_count: Dict[str, int] = {}
            y_by_clip: Dict[str, int] = {}

            for rp in all_preds:
                for cid, yt, _yp, pr in rp:
                    prob_sum[cid] = prob_sum.get(cid, 0.0) + pr
                    prob_count[cid] = prob_count.get(cid, 0) + 1
                    y_by_clip[cid] = yt

            clip_id_list = list(clip_ids)
            probs_avg = [prob_sum[cid] / max(1, prob_count[cid]) for cid in clip_id_list]
            y_true = [y_by_clip[cid] for cid in clip_id_list]
            auc = float(roc_auc_score(y_true, probs_avg))
    except Exception:
        auc = float("nan")

    label = getattr(clf_factory, "label", None) or getattr(clf_factory, "__name__", None) or clf_factory.__class__.__name__
    print(f"[{label}] Clip-LOOCV  acc={mean_acc:.4f} ± {std_acc:.4f}  auc={auc:.4f}")

    return mean_acc, std_acc, all_preds[0]


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
    if pd is None:
        raise ModuleNotFoundError("pandas is required for build_dataset() / RandomForest CV path")
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
    if pd is None:
        raise ModuleNotFoundError("pandas is required for run_cv_and_train() / RandomForest CV path")
    if Pipeline is None or cross_val_predict is None or StandardScaler is None:
        raise ModuleNotFoundError("scikit-learn is required for run_cv_and_train() / RandomForest CV path")
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
    if joblib is None:
        raise ModuleNotFoundError(
            "joblib is required for saving the fitted RandomForest model in run_cv_and_train()."
        )
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
    p.add_argument(
        "--feature-csvs",
        nargs="*",
        default=None,
        help="Optional: if provided, run subject-aware LOOCV on the aligned clip-level feature CSVs.",
    )
    p.add_argument("--n-runs", type=int, default=3, help="Number of repeated subject-LOOCV runs")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    out_dir = (root / args.out).resolve()

    print(f"DATA DIR: {data_dir}")
    print(f"OUTPUT: {out_dir}")
    if args.feature_csvs:
        print("Running subject-aware Clip-LOOCV...")
        if RandomForestClassifier is None:
            raise ModuleNotFoundError("scikit-learn is required for subject-aware LOOCV")
        X, y, subject_ids, clip_ids = load_features([str(Path(c).resolve()) for c in args.feature_csvs])

        def rf_factory(run_seed: int):
            return RandomForestClassifier(
                n_estimators=200,
                random_state=run_seed,
                class_weight="balanced",
                n_jobs=-1,
            )

        # Attach a label used in print formatting.
        setattr(rf_factory, "label", "RF")

        mean_acc, std_acc, first_run_preds = run_loocv(
            X=X,
            y=y,
            subject_ids=subject_ids,
            clip_ids=clip_ids,
            clf_factory=rf_factory,
            scaler=True,
            n_runs=args.n_runs,
        )
        _ = first_run_preds  # currently unused; returned for possible downstream AUC/confusion analysis
        print(f"Saved LOOCV summary to {out_dir} (metrics printed above)")
    else:
        print("Building dataset (this may take a moment)...")
        Xdf, y = build_dataset(data_dir, conf_thr=args.conf_thr)
        print(f"Found {len(Xdf)} trials. Feature vector size: {Xdf.shape[1]} (including metadata)")

        print("Running CV + training RandomForest...")
        results = run_cv_and_train(
            Xdf, y, out_dir, n_splits=args.n_splits, random_state=args.random_state
        )

        print("Results:")
        print(json.dumps(results, indent=2))
        print(f"Saved model+features+results to {out_dir}")

if __name__ == "__main__":
    main()
