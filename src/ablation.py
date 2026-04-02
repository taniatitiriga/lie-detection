from __future__ import annotations

import csv
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
    vis_path: str,
    acou_path: str,
    ling_path: str,
    n_runs: int = 3,
    out_dir: str | None = None,
    subject_level: bool = False,
) -> list[dict]:
    """
    Run the full ablation table (single-modality, early fusion, late fusion)
    and return a list of result dicts.

    When subject_level=True an extra block of subject-level majority-vote rows
    is appended to the returned list (and saved to the CSV under eval_level='subject').
    """
    # Load each modality via load_features so clip_id alignment is enforced.
    # Use the visual CSV's ordering as the canonical clip order, then reindex
    # acoustic and linguistic to match it.
    X_vis, y, subject_ids, clip_ids = load_features([vis_path])

    X_acou, _, _, acou_clip_ids = load_features([acou_path])
    acou_idx = {cid: i for i, cid in enumerate(acou_clip_ids)}
    X_acou = X_acou[[acou_idx[cid] for cid in clip_ids]]

    X_ling, _, _, ling_clip_ids = load_features([ling_path])
    ling_idx = {cid: i for i, cid in enumerate(ling_clip_ids)}
    X_ling = X_ling[[ling_idx[cid] for cid in clip_ids]]

    modality_X = {
        "Visual": X_vis,
        "Acoustic": X_acou,
        "Linguistic": X_ling,
    }

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
    for mod_name, X_mod in modality_X.items():
        for factory in [rf_factory, svm_factory, nn_factory]:
            print(f"\n--- {mod_name} / {factory.label} ---")
            acc, std, _, auc, s_acc, s_std = run_loocv(
                X_mod, y, subject_ids, clip_ids, factory,
                scaler=True, n_runs=n_runs, subject_level=subject_level,
            )
            _record(mod_name, "none", factory.label, acc, std, auc, eval_level="clip")

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
        acc, std, _, auc, s_acc, s_std = run_loocv(
            X_early, y, subject_ids, clip_ids, nn_factory,
            scaler=True, n_runs=n_runs, subject_level=subject_level,
        )
        _record(combo_name, "early", "NN", acc, std, auc, eval_level="clip")

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
        _record(combo_name, "late", "NN", acc, std, auc, eval_level="clip", best_w=w)

    # --- 4. All three — early fusion (RF, SVM, NN) ---
    print("\n===== All three — early fusion =====")
    X_all_early = np.hstack([X_vis, X_acou, X_ling])
    for factory in [rf_factory, svm_factory, nn_factory]:
        print(f"\n--- All / early / {factory.label} ---")
        acc, std, _, auc, s_acc, s_std = run_loocv(
            X_all_early, y, subject_ids, clip_ids, factory,
            scaler=True, n_runs=n_runs, subject_level=subject_level,
        )
        _record("All", "early", factory.label, acc, std, auc, eval_level="clip")

    # --- 5. All three — late fusion (RF, NN) ---
    print("\n===== All three — late fusion =====")
    for factory in [rf_factory, nn_factory]:
        print(f"\n--- All / late / {factory.label} ---")
        acc, std, auc, w = run_late_fusion_loocv(
            [X_vis, X_acou, X_ling], y, subject_ids, clip_ids,
            factory, n_runs=n_runs,
        )
        _record("All", "late", factory.label, acc, std, auc, eval_level="clip", best_w=w)

    # --- Change 3: Subject-level majority-vote rows ---
    if subject_level:
        print("\n===== Subject-level majority-vote accuracy =====")
        subject_ids_arr = np.array(subject_ids)
        clip_id_list = list(clip_ids)

        def _subj_level_acc_for(
            X_m: np.ndarray, factory, n_runs: int
        ) -> tuple[float, float]:
            """Return (mean_subj_acc, std_subj_acc) across runs."""
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

        for mod_name, X_mod in modality_X.items():
            for factory in [rf_factory, svm_factory, nn_factory]:
                s_acc, s_std = _subj_level_acc_for(X_mod, factory, n_runs)
                print(f"[{factory.label}] {mod_name} Subj-acc={s_acc:.4f} ± {s_std:.4f}")
                _record(mod_name, "none", factory.label,
                        s_acc, s_std, float("nan"), eval_level="subject")

        # All-three early fusion
        for factory in [rf_factory, svm_factory, nn_factory]:
            s_acc, s_std = _subj_level_acc_for(X_all_early, factory, n_runs)
            print(f"[{factory.label}] All/early Subj-acc={s_acc:.4f} ± {s_std:.4f}")
            _record("All", "early", factory.label,
                    s_acc, s_std, float("nan"), eval_level="subject")

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