#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import json

from data_loader import load_features
from models import rf_factory, svm_factory, nn_factory
from evaluation import run_sanity_checks, run_loocv
from ablation import run_ablation

try:
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    RandomForestClassifier = None  # type: ignore[assignment]


def main():
    p = argparse.ArgumentParser(
        description="Lie-detection LOSO-CV classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Subject-aware LOOCV on pre-built feature CSVs (RF + SVM + NN):
  uv run python src/main.py \\
      --feature-csvs features/facial.csv features/visual.csv features/acoustic.csv features/linguistic.csv \\
      --n-runs 3

  # Same, also report subject-level majority-vote accuracy:
  uv run python src/main.py \\
      --feature-csvs features/facial.csv features/visual.csv features/acoustic.csv features/linguistic.csv \\
      --n-runs 3 --subject-level

  # Full ablation table saved to runs/my_experiment/clip_level_results.csv:
  uv run python src/main.py --ablation --n-runs 3 --out runs/my_experiment

  # Full ablation + subject-level majority-vote rows:
  uv run python src/main.py --ablation --subject-level --n-runs 3 --out runs/my_experiment

  # Sanity checks (leakage, scaler, dummy baseline):
  uv run python src/main.py --sanity
"""
    )
    p.add_argument("--out", type=str, default="runs/experiment", help="Output folder to save model/results")
    p.add_argument(
        "--feature-csvs",
        nargs="*",
        default=None,
        help="One or more pre-built clip-level feature CSVs (inner-joined on clip_id)."
             " Runs subject-aware LOSO-CV with RF, SVM, and NN.",
    )
    p.add_argument("--n-runs", type=int, default=3, help="Number of repeated subject-LOOCV runs (reduces variance)")
    p.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run full ablation table (single-modality, early+late fusion, all classifiers)."
             " Reads features/facial.csv, features/visual.csv, features/acoustic.csv, features/linguistic.csv."
             " Saves clip_level_results.csv to --out.",
    )
    p.add_argument(
        "--subject-level",
        action="store_true",
        default=False,
        help="Also compute and report subject-level majority-vote accuracy (mean posterior"
             " per subject → binary prediction). In --ablation mode the subject-level rows"
             " are written to the results CSV under eval_level='subject'.",
    )
    p.add_argument(
        "--sanity",
        action="store_true",
        default=False,
        help="Run sanity checks (subject leakage, scaler leakage, dummy baseline).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out).resolve()

    print(f"OUTPUT: {out_dir}")

    if args.sanity:
        feat_dir = root / "features"
        run_sanity_checks(
            vis_path=str((feat_dir / "visual.csv").resolve()),
            fac_path=str((feat_dir / "facial.csv").resolve()),
            acou_path=str((feat_dir / "acoustic.csv").resolve()),
            ling_path=str((feat_dir / "linguistic.csv").resolve()),
            n_runs=args.n_runs,
        )

    elif args.ablation:
        if RandomForestClassifier is None:
            raise ModuleNotFoundError("scikit-learn is required for ablation")
        feat_dir = root / "features"
        fac_path = str((feat_dir / "facial.csv").resolve())
        acou_path = str((feat_dir / "acoustic.csv").resolve())
        ling_path = str((feat_dir / "linguistic.csv").resolve())
        print(f"Running full ablation (n_runs={args.n_runs}) ...")
        run_ablation(
            fac_path=fac_path,
            acou_path=acou_path,
            ling_path=ling_path,
            n_runs=args.n_runs,
            out_dir=str(out_dir),
            subject_level=args.subject_level,
        )

    elif args.feature_csvs:
        import csv
        print("Running subject-aware Clip-LOOCV (macro-averaged fold accuracy)...")
        if RandomForestClassifier is None:
            raise ModuleNotFoundError("scikit-learn is required for subject-aware LOOCV")
        X, y, subject_ids, clip_ids = load_features([str(Path(c).resolve()) for c in args.feature_csvs])

        results = []
        for factory in [rf_factory, svm_factory, nn_factory]:
            mean_acc, std_acc, _, auc, subj_mean, subj_std = run_loocv(
                X=X,
                y=y,
                subject_ids=subject_ids,
                clip_ids=clip_ids,
                clf_factory=factory,
                scaler=True,
                n_runs=args.n_runs,
                subject_level=args.subject_level,
            )
            classifier_label = getattr(factory, "label", getattr(factory, "__name__", "CLF"))
            results.append({
                "eval_level": "clip",
                "classifier": classifier_label,
                "mean_acc": round(mean_acc, 4),
                "std_acc": round(std_acc, 4),
                "auc": round(auc, 4) if auc == auc else float("nan"),
            })
            if args.subject_level:
                results.append({
                    "eval_level": "subject",
                    "classifier": classifier_label,
                    "mean_acc": round(subj_mean, 4),
                    "std_acc": round(subj_std, 4),
                    "auc": float("nan"),
                })
        
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "custom_features_results.csv"
        cols = ["eval_level", "classifier", "mean_acc", "std_acc", "auc"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
        print(f"\nResults saved to {csv_path}")
    else:
        p.print_help()

if __name__ == "__main__":
    main()