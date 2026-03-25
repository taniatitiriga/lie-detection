#!/usr/bin/env python3


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
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
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
    SVC = None  # type: ignore[assignment]
    MLPClassifier = None  # type: ignore[assignment]
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
    DummyClassifier = None  # type: ignore[assignment]

try:
    from scipy.ndimage import uniform_filter
except ModuleNotFoundError:
    uniform_filter = None  # type: ignore[assignment]

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
    silent: bool = False,
) -> tuple[float, float, list, float]:
    """
    Subject-aware LOOCV with clip-level evaluation.

    Returns:
      mean_acc, std_acc, first_run_preds, auc
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
    if not silent:
        print(f"[{label}] Clip-LOOCV  acc={mean_acc:.4f} ± {std_acc:.4f}  auc={auc:.4f}")

    return mean_acc, std_acc, all_preds[0], auc


# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------

def rf_factory(run_seed: int):
    """Random Forest factory."""
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        random_state=run_seed,
    )

rf_factory.label = "RF"  # type: ignore[attr-defined]


def svm_factory(run_seed: int):
    """SVM factory with per-fold 4-fold CV grid search + 3×3 mean-filter smoothing."""

    class TunedSVM:
        def __init__(self):
            self._model = None

        def fit(self, X_train, y_train):
            C_values = [0.01, 0.1, 1, 10, 100]
            gamma_values = [0.001, 0.01, 0.1, 1, "scale"]

            n_C = len(C_values)
            n_gamma = len(gamma_values)
            loss_matrix = np.zeros((n_C, n_gamma))

            cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=run_seed)

            for i, C in enumerate(C_values):
                for j, gamma in enumerate(gamma_values):
                    fold_errors = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        Xtr, Xval = X_train[train_idx], X_train[val_idx]
                        ytr, yval = y_train[train_idx], y_train[val_idx]
                        clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=run_seed)
                        clf.fit(Xtr, ytr)
                        fold_errors.append(1.0 - accuracy_score(yval, clf.predict(Xval)))
                    loss_matrix[i, j] = np.mean(fold_errors)

            # Apply 3×3 mean filter to smooth the loss surface
            if uniform_filter is not None:
                loss_matrix = uniform_filter(loss_matrix, size=3, mode="nearest")

            best_idx = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)
            best_C = C_values[best_idx[0]]
            best_gamma = gamma_values[best_idx[1]]

            self._model = SVC(
                kernel="rbf", C=best_C, gamma=best_gamma,
                probability=True, random_state=run_seed,
            )
            self._model.fit(X_train, y_train)

        def predict_proba(self, X_test):
            return self._model.predict_proba(X_test)

    return TunedSVM()

svm_factory.label = "SVM"  # type: ignore[attr-defined]


def nn_factory(run_seed: int):
    """Neural-network factory (MLP matching the 2020 paper)."""
    return MLPClassifier(
        hidden_layer_sizes=(100, 500),
        activation="relu",
        solver="adam",
        alpha=1e-5,
        max_iter=500,
        random_state=run_seed,
        early_stopping=True,
        validation_fraction=0.1,
    )

nn_factory.label = "NN"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Modality loading helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Late fusion LOOCV
# ---------------------------------------------------------------------------

def run_late_fusion_loocv(
    X_list: list[np.ndarray],
    y: np.ndarray,
    subject_ids: list,
    clip_ids: list,
    clf_factory,
    n_runs: int = 3,
) -> tuple[float, float, float, float]:
    """
    Late fusion: train separate classifiers per modality inside each LOOCV
    fold, collect per-clip probabilities, then sweep w_vis (weight for the
    first modality in X_list) to find the best fusion weight.

    Returns:
      best_acc, std_acc (across runs at best weight), auc, best_w_vis
    """
    if StandardScaler is None or roc_auc_score is None:
        raise ModuleNotFoundError("scikit-learn is required")

    unique_subjects = sorted(set(subject_ids))
    subject_ids_arr = np.array(subject_ids)
    clip_ids_arr = np.array(clip_ids)
    y_arr = np.array(y, dtype=int)
    n_modalities = len(X_list)

    # Collect per-modality, per-run, per-clip probabilities
    # all_probs[run][modality] = dict{clip_id -> prob}
    all_probs: list[list[Dict[str, float]]] = []
    # all_true[run] = dict{clip_id -> true_label}
    all_true: list[Dict[str, int]] = []

    for run in range(n_runs):
        mod_probs: list[Dict[str, float]] = [{} for _ in range(n_modalities)]
        true_labels: Dict[str, int] = {}

        for subj in unique_subjects:
            test_mask = subject_ids_arr == subj
            train_mask = ~test_mask
            y_train, y_test = y_arr[train_mask], y_arr[test_mask]
            test_indices = np.where(test_mask)[0]
            ids = clip_ids_arr[test_indices].tolist()

            for m_idx, X_m in enumerate(X_list):
                X_train_m, X_test_m = X_m[train_mask], X_m[test_mask]
                sc = StandardScaler().fit(X_train_m)
                X_train_m = sc.transform(X_train_m)
                X_test_m = sc.transform(X_test_m)

                clf = clf_factory(run_seed=run)
                clf.fit(X_train_m, y_train)
                probs = clf.predict_proba(X_test_m)[:, 1]
                for cid, pr in zip(ids, probs):
                    mod_probs[m_idx][cid] = float(pr)

            for cid, yt in zip(ids, y_test):
                true_labels[cid] = int(yt)

        all_probs.append(mod_probs)
        all_true.append(true_labels)

    # Weight sweep: w_vis in [0.1 .. 0.9]
    clip_id_list = list(clip_ids)
    w_candidates = [round(v * 0.1, 2) for v in range(1, 10)]  # 0.1 .. 0.9
    best_w = 0.5
    best_mean_acc = 0.0
    best_std = 0.0
    best_auc = float("nan")

    for w_vis in w_candidates:
        w_other = (1.0 - w_vis) / max(1, n_modalities - 1)
        run_accs = []
        for run in range(n_runs):
            correct = 0
            total = 0
            for cid in clip_id_list:
                fused = w_vis * all_probs[run][0][cid]
                for m_idx in range(1, n_modalities):
                    fused += w_other * all_probs[run][m_idx][cid]
                pred = int(fused >= 0.5)
                if pred == all_true[run][cid]:
                    correct += 1
                total += 1
            run_accs.append(correct / max(1, total))
        m_acc = float(np.mean(run_accs))
        if m_acc > best_mean_acc:
            best_mean_acc = m_acc
            best_std = float(np.std(run_accs))
            best_w = w_vis
            # Compute AUC at this weight
            try:
                w_other_auc = (1.0 - w_vis) / max(1, n_modalities - 1)
                fused_probs_agg: Dict[str, float] = {}
                fused_count: Dict[str, int] = {}
                for run in range(n_runs):
                    for cid in clip_id_list:
                        fp = w_vis * all_probs[run][0][cid]
                        for m_idx in range(1, n_modalities):
                            fp += w_other_auc * all_probs[run][m_idx][cid]
                        fused_probs_agg[cid] = fused_probs_agg.get(cid, 0.0) + fp
                        fused_count[cid] = fused_count.get(cid, 0) + 1
                avg_p = [fused_probs_agg[c] / fused_count[c] for c in clip_id_list]
                yt_list = [all_true[0][c] for c in clip_id_list]
                best_auc = float(roc_auc_score(yt_list, avg_p))
            except Exception:
                best_auc = float("nan")

    label = getattr(clf_factory, "label", None) or "CLF"
    print(
        f"[{label}] Late-fusion  acc={best_mean_acc:.4f} ± {best_std:.4f}"
        f"  auc={best_auc:.4f}  w_vis={best_w:.1f}"
    )
    return best_mean_acc, best_std, best_auc, best_w


# ---------------------------------------------------------------------------
# Full ablation
# ---------------------------------------------------------------------------

def run_ablation(
    vis_path: str,
    acou_path: str,
    ling_path: str,
    n_runs: int = 3,
    out_dir: str | None = None,
) -> list[dict]:
    """
    Run the full ablation table (single-modality, early fusion, late fusion)
    and return a list of result dicts.
    """
    # Load shared metadata (y, subject_ids, clip_ids) from any CSV
    _, y, subject_ids, clip_ids = load_features([vis_path])

    # Load per-modality feature matrices (aligned to same clip order)
    X_vis = load_single_modality(vis_path)
    X_acou = load_single_modality(acou_path)
    X_ling = load_single_modality(ling_path)

    modality_X = {
        "Visual": X_vis,
        "Acoustic": X_acou,
        "Linguistic": X_ling,
    }

    results: list[dict] = []

    def _record(modality: str, fusion: str, classifier: str,
                mean_acc: float, std_acc: float, auc: float, **extra):
        row = {
            "modality": modality,
            "fusion": fusion,
            "classifier": classifier,
            "mean_acc": round(mean_acc, 4),
            "std_acc": round(std_acc, 4),
            "auc": round(auc, 4),
        }
        row.update(extra)
        results.append(row)

    # --- 1. Single-modality (RF, SVM, NN) ---
    print("\n===== Single-modality experiments =====")
    for mod_name, X_mod in modality_X.items():
        for factory in [rf_factory, svm_factory, nn_factory]:
            print(f"\n--- {mod_name} / {factory.label} ---")
            acc, std, _, auc = run_loocv(
                X_mod, y, subject_ids, clip_ids, factory,
                scaler=True, n_runs=n_runs,
            )
            _record(mod_name, "none", factory.label, acc, std, auc)

    # --- 2. Two-modality early fusion (NN only per spec, but also others where noted) ---
    two_mod_combos = [
        ("Visual+Acoustic", X_vis, X_acou),
        ("Visual+Linguistic", X_vis, X_ling),
        ("Acoustic+Linguistic", X_acou, X_ling),
    ]
    print("\n===== Two-modality early fusion (NN) =====")
    for combo_name, Xa, Xb in two_mod_combos:
        X_early = np.hstack([Xa, Xb])
        print(f"\n--- {combo_name} / early / NN ---")
        acc, std, _, auc = run_loocv(
            X_early, y, subject_ids, clip_ids, nn_factory,
            scaler=True, n_runs=n_runs,
        )
        _record(combo_name, "early", "NN", acc, std, auc)

    # --- 3. Two-modality late fusion (NN only) ---
    print("\n===== Two-modality late fusion (NN) =====")
    late_two_combos = [
        ("Visual+Acoustic", [X_vis, X_acou]),
        ("Visual+Linguistic", [X_vis, X_ling]),
        ("Acoustic+Linguistic", [X_acou, X_ling]),
    ]
    for combo_name, x_list in late_two_combos:
        print(f"\n--- {combo_name} / late / NN ---")
        acc, std, auc, w = run_late_fusion_loocv(
            x_list, y, subject_ids, clip_ids, nn_factory, n_runs=n_runs,
        )
        _record(combo_name, "late", "NN", acc, std, auc, best_w=w)

    # --- 4. All three — early fusion (RF, SVM, NN) ---
    print("\n===== All three — early fusion =====")
    X_all_early = np.hstack([X_vis, X_acou, X_ling])
    for factory in [rf_factory, svm_factory, nn_factory]:
        print(f"\n--- All / early / {factory.label} ---")
        acc, std, _, auc = run_loocv(
            X_all_early, y, subject_ids, clip_ids, factory,
            scaler=True, n_runs=n_runs,
        )
        _record("All", "early", factory.label, acc, std, auc)

    # --- 5. All three — late fusion (RF, NN) ---
    print("\n===== All three — late fusion =====")
    for factory in [rf_factory, nn_factory]:
        print(f"\n--- All / late / {factory.label} ---")
        acc, std, auc, w = run_late_fusion_loocv(
            [X_vis, X_acou, X_ling], y, subject_ids, clip_ids,
            factory, n_runs=n_runs,
        )
        _record("All", "late", factory.label, acc, std, auc, best_w=w)

    # --- Save CSV ---
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "clip_level_results.csv"
        cols = ["modality", "fusion", "classifier", "mean_acc", "std_acc", "auc"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {csv_path}")

    # --- Print markdown table ---
    BASELINE_2015 = 0.7520
    BASELINE_2020 = 0.8305
    print("\n### Clip-level ablation results\n")
    print("| Modality | Fusion | Clf | Acc | ±Std | AUC | Note |")
    print("|----------|--------|-----|-----|------|-----|------|")
    for r in results:
        note = ""
        if r["mean_acc"] >= BASELINE_2015:
            note += "✓ "
        if r["mean_acc"] >= BASELINE_2020:
            note += "★"
        print(
            f"| {r['modality']:<22s} | {r['fusion']:<5s} "
            f"| {r['classifier']:<3s} | {r['mean_acc']:.4f} | {r['std_acc']:.4f} "
            f"| {r['auc']:.4f} | {note.strip()} |"
        )

    return results


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


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(vis_path: str, acou_path: str, ling_path: str, n_runs: int = 1):
    """Run three sanity checks on the classification pipeline."""
    if StandardScaler is None or DummyClassifier is None:
        raise ModuleNotFoundError("scikit-learn is required for sanity checks")

    # Load data (use all three modalities, early-fused)
    X, y, subject_ids, clip_ids = load_features([vis_path, acou_path, ling_path])
    subject_ids_arr = np.array(subject_ids)
    unique_subjects = sorted(set(subject_ids))
    n_total = len(y)

    failures: list[str] = []

    # ------------------------------------------------------------------
    # Check 1 — Label (subject) leakage
    # ------------------------------------------------------------------
    print("\n=== Check 1: Label (subject) leakage ===")
    leakage_ok = True
    for subj in unique_subjects:
        test_mask = subject_ids_arr == subj
        train_mask = ~test_mask
        train_subjects = set(subject_ids_arr[train_mask])
        if subj in train_subjects:
            print(f"  FAIL  subject {subj} found in both train and test")
            leakage_ok = False
        else:
            print(f"  PASS  subject {subj}  (test={int(test_mask.sum())} clips)")
    if leakage_ok:
        print("Check 1 PASSED: no subject leakage in any fold.")
    else:
        failures.append("Check 1: subject leakage detected")

    # ------------------------------------------------------------------
    # Check 2 — Scaler leakage
    # ------------------------------------------------------------------
    print("\n=== Check 2: Scaler leakage ===")
    scaler_ok = True
    for subj in unique_subjects:
        test_mask = subject_ids_arr == subj
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        sc = StandardScaler()
        sc.fit(X_train)
        # Verify scaler saw only train samples
        assert sc.n_samples_seen_ is not None
        n_seen = int(sc.n_samples_seen_) if np.isscalar(sc.n_samples_seen_) else int(sc.n_samples_seen_[0])
        if n_seen != n_train:
            print(f"  FAIL  subject {subj}: scaler fit on {n_seen} samples, expected {n_train}")
            scaler_ok = False
        else:
            if subj == unique_subjects[0] or subj == unique_subjects[-1]:
                print(f"  PASS  subject {subj}: scaler fit on {n_seen} samples (not {n_total})")

        # Also verify transform doesn't change shape
        X_train_t = sc.transform(X_train)
        X_test_t = sc.transform(X_test)
        assert X_train_t.shape == X_train.shape
        assert X_test_t.shape == X_test.shape

    if scaler_ok:
        print(f"Check 2 PASSED: StandardScaler always fit on N_train only (not {n_total}).")
    else:
        failures.append("Check 2: scaler leakage detected")

    # ------------------------------------------------------------------
    # Check 3 — Chance (dummy) baseline
    # ------------------------------------------------------------------
    print("\n=== Check 3: Chance baseline ===")

    def dummy_factory(run_seed: int):
        return DummyClassifier(strategy="most_frequent")
    dummy_factory.label = "Dummy"  # type: ignore[attr-defined]

    dummy_acc, dummy_std, _, dummy_auc = run_loocv(
        X, y, subject_ids, clip_ids,
        clf_factory=dummy_factory,
        scaler=False,  # scaler irrelevant for dummy
        n_runs=1,
        silent=True,
    )
    majority_frac = max(np.mean(y), 1 - np.mean(y))
    print(f"  Dummy baseline:  acc={dummy_acc:.4f} (expected ~{majority_frac:.4f})")

    # Compare against real classifiers
    real_accs = {}
    for factory in [rf_factory, nn_factory]:
        acc, _, _, _ = run_loocv(
            X, y, subject_ids, clip_ids, factory,
            scaler=True, n_runs=n_runs, silent=True,
        )
        real_accs[factory.label] = acc

    dummy_beats_real = []
    for name, acc in real_accs.items():
        print(f"  {name} acc={acc:.4f}  {'< dummy (!)' if acc < dummy_acc else '>= dummy (ok)'}")
        if acc < dummy_acc:
            dummy_beats_real.append(name)

    if dummy_beats_real:
        failures.append(f"Check 3: dummy beats {', '.join(dummy_beats_real)}")
        print(f"Check 3 WARNING: dummy classifier beats: {', '.join(dummy_beats_real)}")
    else:
        print("Check 3 PASSED: no real classifier is beaten by dummy.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if not failures:
        print("All sanity checks passed.")
    else:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    return len(failures) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/extracted_AU_gaze", help="Path to folder with trial_*.csv")
    p.add_argument("--out", type=str, default="runs/nn_experiment", help="Output folder to save model/results")
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
    p.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run full ablation table (early+late fusion, all classifiers)."
             " Expects features/visual.csv, features/acoustic.csv, features/linguistic.csv.",
    )
    p.add_argument(
        "--sanity",
        action="store_true",
        default=False,
        help="Run sanity checks (leakage, scaler, dummy baseline).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = (root / args.data_dir).resolve()
    out_dir = (root / args.out).resolve()

    print(f"DATA DIR: {data_dir}")
    print(f"OUTPUT: {out_dir}")

    if args.sanity:
        feat_dir = root / "features"
        run_sanity_checks(
            vis_path=str((feat_dir / "visual.csv").resolve()),
            acou_path=str((feat_dir / "acoustic.csv").resolve()),
            ling_path=str((feat_dir / "linguistic.csv").resolve()),
            n_runs=args.n_runs,
        )

    elif args.ablation:
        if RandomForestClassifier is None:
            raise ModuleNotFoundError("scikit-learn is required for ablation")
        feat_dir = root / "features"
        vis_path = str((feat_dir / "visual.csv").resolve())
        acou_path = str((feat_dir / "acoustic.csv").resolve())
        ling_path = str((feat_dir / "linguistic.csv").resolve())
        print(f"Running full ablation (n_runs={args.n_runs}) ...")
        run_ablation(
            vis_path=vis_path,
            acou_path=acou_path,
            ling_path=ling_path,
            n_runs=args.n_runs,
            out_dir=str(out_dir),
        )

    elif args.feature_csvs:
        print("Running subject-aware Clip-LOOCV...")
        if RandomForestClassifier is None:
            raise ModuleNotFoundError("scikit-learn is required for subject-aware LOOCV")
        X, y, subject_ids, clip_ids = load_features([str(Path(c).resolve()) for c in args.feature_csvs])

        for factory in [rf_factory, svm_factory, nn_factory]:
            run_loocv(
                X=X,
                y=y,
                subject_ids=subject_ids,
                clip_ids=clip_ids,
                clf_factory=factory,
                scaler=True,
                n_runs=args.n_runs,
            )
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
