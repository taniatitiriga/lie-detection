from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
import numpy as np
from typing import Dict

try:
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    StandardScaler = None  # type: ignore[assignment]

from data_loader import load_features
from models import rf_factory, svm_factory, nn_factory
from evaluation import run_loocv, run_late_fusion_loocv


def run_ablation(
    fac_path: str,
    vis_path: str,
    acou_path: str,
    ling_path: str,
    n_runs: int = 3,
    out_dir: str | None = None,
    subject_level: bool = False,
) -> list[dict]:
    """
    Run the full ablation table (single-modality, pairwise fusion, all-modality
    fusion) across 4 modalities and return a list of result dicts.
    """
    # Load each modality; use the facial CSV's ordering as canonical clip order.
    X_fac, y, subject_ids, clip_ids = load_features([fac_path])

    def _reindex(path: str) -> np.ndarray:
        X, _, _, cids = load_features([path])
        idx = {cid: i for i, cid in enumerate(cids)}
        return X[[idx[cid] for cid in clip_ids]]

    X_vis = _reindex(vis_path)
    X_acou = _reindex(acou_path)
    X_ling = _reindex(ling_path)

    modality_X: dict[str, np.ndarray] = {
        "Facial": X_fac,
        "Visual": X_vis,
        "Acoustic": X_acou,
        "Linguistic": X_ling,
    }
    mod_names = list(modality_X.keys())

    results: list[dict] = []

    def _record(modality: str, fusion: str, classifier: str,
                mean_acc: float, std_acc: float, auc: float,
                eval_level: str = "clip", **extra):
        row = {
            "eval_level": eval_level,
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
    for mod_name in mod_names:
        for factory in [rf_factory, svm_factory, nn_factory]:
            print(f"\n--- {mod_name} / {factory.label} ---")
            acc, std, _, auc, s_acc, s_std = run_loocv(
                modality_X[mod_name], y, subject_ids, clip_ids, factory,
                scaler=True, n_runs=n_runs, subject_level=subject_level,
            )
            _record(mod_name, "none", factory.label, acc, std, auc, eval_level="clip")

    # --- 2. Two-modality early fusion (SVM + NN) ---
    print("\n===== Two-modality early fusion (SVM, NN) =====")
    for a, b in combinations(mod_names, 2):
        combo_name = f"{a}+{b}"
        X_early = np.hstack([modality_X[a], modality_X[b]])
        for factory in [svm_factory, nn_factory]:
            print(f"\n--- {combo_name} / early / {factory.label} ---")
            acc, std, _, auc, s_acc, s_std = run_loocv(
                X_early, y, subject_ids, clip_ids, factory,
                scaler=True, n_runs=n_runs, subject_level=subject_level,
            )
            _record(combo_name, "early", factory.label, acc, std, auc, eval_level="clip")
            if subject_level:
                _record(combo_name, "early", factory.label, s_acc, s_std, float("nan"), eval_level="subject")

    # --- 3. Two-modality late fusion (SVM + NN) ---
    print("\n===== Two-modality late fusion (SVM, NN) =====")
    for a, b in combinations(mod_names, 2):
        combo_name = f"{a}+{b}"
        x_list = [modality_X[a], modality_X[b]]
        for factory in [svm_factory, nn_factory]:
            print(f"\n--- {combo_name} / late / {factory.label} ---")
            acc, std, auc, w = run_late_fusion_loocv(
                x_list, y, subject_ids, clip_ids, factory, n_runs=n_runs,
            )
            _record(combo_name, "late", factory.label, acc, std, auc, eval_level="clip", best_w=w)

    # --- 4. All four — early fusion (RF, SVM, NN) ---
    print("\n===== All four — early fusion =====")
    X_all_early = np.hstack([modality_X[m] for m in mod_names])
    for factory in [rf_factory, svm_factory, nn_factory]:
        print(f"\n--- All / early / {factory.label} ---")
        acc, std, _, auc, s_acc, s_std = run_loocv(
            X_all_early, y, subject_ids, clip_ids, factory,
            scaler=True, n_runs=n_runs, subject_level=subject_level,
        )
        _record("All", "early", factory.label, acc, std, auc, eval_level="clip")
        if subject_level:
            _record("All", "early", factory.label, s_acc, s_std, float("nan"), eval_level="subject")

    # --- 5. All four — late fusion (RF, SVM, NN) ---
    print("\n===== All four — late fusion =====")
    x_all_list = [modality_X[m] for m in mod_names]
    for factory in [rf_factory, svm_factory, nn_factory]:
        print(f"\n--- All / late / {factory.label} ---")
        acc, std, auc, w = run_late_fusion_loocv(
            x_all_list, y, subject_ids, clip_ids,
            factory, n_runs=n_runs,
        )
        _record("All", "late", factory.label, acc, std, auc, eval_level="clip", best_w=w)

    # --- Subject-level majority-vote rows ---
    if subject_level:
        print("\n===== Subject-level majority-vote accuracy =====")
        subject_ids_arr = np.array(subject_ids)
        clip_id_list = list(clip_ids)

        def _subj_level_acc_for(
            X_m: np.ndarray, factory, n_runs: int
        ) -> tuple[float, float]:
            unique_subjects = sorted(set(subject_ids))
            run_accs: list[float] = []
            for run in range(n_runs):
                subj_prob_sum: Dict[str, float] = {}
                subj_prob_cnt: Dict[str, int] = {}
                subj_true: Dict[str, int] = {}
                for subj in unique_subjects:
                    test_mask = subject_ids_arr == subj
                    train_mask = ~test_mask
                    X_train, X_test = X_m[train_mask], X_m[test_mask]
                    y_train = np.array(y)[train_mask]
                    y_test = np.array(y)[test_mask]
                    sc = StandardScaler().fit(X_train)
                    X_train = sc.transform(X_train)
                    X_test = sc.transform(X_test)
                    clf = factory(run_seed=run)
                    clf.fit(X_train, y_train)
                    probs = clf.predict_proba(X_test)[:, 1]
                    test_indices = np.where(test_mask)[0]
                    ids = [clip_id_list[i] for i in test_indices]
                    for cid, pr, yt in zip(ids, probs, y_test):
                        subj_prob_sum[subj] = subj_prob_sum.get(subj, 0.0) + pr
                        subj_prob_cnt[subj] = subj_prob_cnt.get(subj, 0) + 1
                        subj_true[subj] = int(yt)
                correct = sum(
                    int((subj_prob_sum[s] / subj_prob_cnt[s] >= 0.5) == subj_true[s])
                    for s in subj_true
                )
                run_accs.append(correct / max(1, len(subj_true)))
            return float(np.mean(run_accs)), float(np.std(run_accs))

        for mod_name in mod_names:
            for factory in [rf_factory, svm_factory, nn_factory]:
                s_acc, s_std = _subj_level_acc_for(modality_X[mod_name], factory, n_runs)
                print(f"[{factory.label}] {mod_name} Subj-acc={s_acc:.4f} ± {s_std:.4f}")
                _record(mod_name, "none", factory.label,
                        s_acc, s_std, float("nan"), eval_level="subject")

        # All/early and two-modality early subject rows are recorded inline above

    # --- Save CSV ---
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "clip_level_results.csv"
        cols = ["eval_level", "modality", "fusion", "classifier", "mean_acc", "std_acc", "auc"]
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
    print("| Level | Modality | Fusion | Clf | Acc | ±Std | AUC | Note |")
    print("|-------|----------|--------|-----|-----|------|-----|------|")
    for r in results:
        note = ""
        if r["mean_acc"] >= BASELINE_2015:
            note += "✓ "
        if r["mean_acc"] >= BASELINE_2020:
            note += "★"
        print(
            f"| {r.get('eval_level', 'clip'):<7s} | {r['modality']:<22s} | {r['fusion']:<5s} "
            f"| {r['classifier']:<3s} | {r['mean_acc']:.4f} | {r['std_acc']:.4f} "
            f"| {r['auc']:.4f} | {note.strip()} |"
        )

    return results
