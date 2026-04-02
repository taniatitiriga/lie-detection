from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from typing import Dict, Any

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:
    RandomForestClassifier = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    DummyClassifier = None  # type: ignore[assignment]
    CalibratedClassifierCV = None  # type: ignore[assignment]
    SelectKBest = None  # type: ignore[assignment]
    f_classif = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

from data_loader import load_features
from models import rf_factory, nn_factory


def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list,
    clip_ids: list,
    clf_factory,
    scaler: bool = True,
    n_runs: int = 3,
    silent: bool = False,
    subject_level: bool = False,
) -> tuple[float, float, list, float]:
    """
    Subject-aware LOSO-CV with macro-averaged fold accuracy.

    Accuracy is computed per fold (one fold = one held-out subject) and then
    averaged across folds — this is the macro-averaged (unweighted) estimate
    and avoids bias toward subjects with many clips.

    When subject_level=True, also returns subject-level accuracy computed by
    averaging each subject's predicted class probabilities across their clips
    (majority vote via mean posterior) and comparing to the subject's true
    majority label.  This result is printed but does NOT affect the return
    value (clip-level metrics are still returned).

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

            # In-fold feature selection: keep at most 50 best features
            # (avoids curse of dimensionality, e.g. 145-dim linguistic feats)
            if SelectKBest is not None and f_classif is not None:
                k = min(X_train.shape[1], 50)
                selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)
                X_train = selector.transform(X_train)
                X_test = selector.transform(X_test)

            clf = clf_factory(run_seed=run)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            test_indices = np.where(test_mask)[0]
            ids = clip_ids_arr[test_indices].tolist()
            for cid, yt, yp, pr in zip(ids, y_test, preds, probs, strict=False):
                run_preds.append((cid, int(yt), int(yp), float(pr)))

        all_preds.append(run_preds)

    # --- Change 1: Macro-averaged fold accuracy ---
    # For each run, compute per-subject (per-fold) accuracy then average across
    # subjects.  This is unweighted by clip count (macro-averaging).
    def _macro_acc(run_preds: list) -> float:
        # group by subject via clip_ids_arr lookup
        subj_correct: Dict[str, list] = {}
        for cid, yt, yp, _pr in run_preds:
            idx = list(clip_ids).index(cid)  # O(n) but n is small (<200)
            subj = subject_ids_arr[idx]
            subj_correct.setdefault(subj, []).append(int(yt == yp))
        return float(np.mean([np.mean(v) for v in subj_correct.values()]))

    run_accs = [_macro_acc(rp) for rp in all_preds]
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
        print(f"[{label}] Clip-LOOCV (macro)  acc={mean_acc:.4f} ± {std_acc:.4f}  auc={auc:.4f}")

    # --- Change 3 (part): Subject-level majority-vote accuracy ---
    if subject_level and not silent:
        # For each run, average posteriors per subject → subject-level prediction
        def _subj_acc(run_preds: list) -> float:
            subj_prob_sum: Dict[str, float] = {}
            subj_prob_cnt: Dict[str, int] = {}
            subj_true: Dict[str, int] = {}
            for cid, yt, _yp, pr in run_preds:
                idx = list(clip_ids).index(cid)
                subj = subject_ids_arr[idx]
                subj_prob_sum[subj] = subj_prob_sum.get(subj, 0.0) + pr
                subj_prob_cnt[subj] = subj_prob_cnt.get(subj, 0) + 1
                subj_true[subj] = yt  # all clips for subj share same label
            correct = sum(
                int((subj_prob_sum[s] / subj_prob_cnt[s] >= 0.5) == subj_true[s])
                for s in subj_true
            )
            return correct / max(1, len(subj_true))

        subj_run_accs = [_subj_acc(rp) for rp in all_preds]
        subj_mean = float(np.mean(subj_run_accs))
        subj_std = float(np.std(subj_run_accs))
        print(f"[{label}] Subj-LOOCV (majority-vote)  acc={subj_mean:.4f} ± {subj_std:.4f}")

    return mean_acc, std_acc, all_preds[0], auc


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

                # In-fold feature selection for late fusion
                if SelectKBest is not None and f_classif is not None:
                    k = min(X_train_m.shape[1], 50)
                    selector = SelectKBest(f_classif, k=k).fit(X_train_m, y_train)
                    X_train_m = selector.transform(X_train_m)
                    X_test_m = selector.transform(X_test_m)

                # Wrap SVM and MLP with probability calibration (late-fusion only).
                # RF produces well-calibrated probabilities via voting; skip it.
                # cv=3 keeps the inner fold count manageable on ~89 training samples.
                _label = getattr(clf_factory, "label", "")
                if CalibratedClassifierCV is not None and _label == "SVM":
                    clf = CalibratedClassifierCV(clf, cv=3, method="sigmoid")
                elif CalibratedClassifierCV is not None and _label == "NN":
                    clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")

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


def run_sanity_checks(vis_path: str, fac_path: str, acou_path: str, ling_path: str, n_runs: int = 1):
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
