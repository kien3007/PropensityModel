"""
Champion Reproduction Check.

Retrains a candidate from this repository (src/config/model_config.py, no promotion),
scores the active champion on the identical out-of-time test rows, and passes when
|AUC(candidate) - AUC(champion)| <= reproduction_auc_tolerance.

Usage:
    python -m scripts.reproduce_champion [--target purchase|lapse] [--data-source lakehouse|csv] [--output-dir DIR]

Exit code 0 = reproduced, 1 = not reproduced.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config.model_config import TRAINING_CONFIGS
from src.features.targets import get_target
from src.config.settings import settings
from src.data.loader import DataLoader
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.models.predictor import PropensityPredictor
from src.pipelines.train_pipeline import run_training_pipeline
from src.training.dataset_builder import DatasetBuilder
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import registry_dir_for

logger = logging.getLogger("ReproduceChampion")

COMPARED_METRICS = ("roc_auc", "pr_auc", "brier_score", "log_loss", "ece", "lift_at_10pct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TRAINING_CONFIGS), default="purchase")
    parser.add_argument("--data-source", choices=["lakehouse", "csv"], default=None)
    parser.add_argument("--output-dir", type=Path, default=settings.OUTPUT_DIR / "reproduction")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = TRAINING_CONFIGS[args.target]
    champion = PropensityPredictor(
        model_path=registry_dir_for(cfg) / "champion_model.joblib",
        expected_horizon_days=cfg.horizon_days,
        expected_target=cfg.target,
    )
    # Reproduce the champion's own recipe (feature set and hyper-parameters it was trained with),
    # which may differ from the current default config.
    recorded_params = champion.bundle.get("lgbm_params")
    cfg = cfg.with_overrides(
        feature_set_version=champion.feature_set_version,
        **({"lgbm_params": dict(recorded_params)} if recorded_params else {}),
    )

    data = DataLoader.load(args.data_source)
    summary = run_training_pipeline(config=cfg, data=data, output_dir=args.output_dir, promote=False)
    candidate = summary["test_metrics"]

    builder = MultiDomainFeatureBuilder(data, lookback_days=cfg.lookback_days,
                                        feature_set_version=cfg.feature_set_version)
    test = DatasetBuilder(builder).build(cfg.test_cutoffs, cfg.horizon_days, get_target(cfg.target))
    incumbent = ModelEvaluator.evaluate(test["target"], champion.predict_proba(test))

    delta_auc = candidate["roc_auc"] - incumbent["roc_auc"]
    reproduced = abs(delta_auc) <= cfg.reproduction_auc_tolerance
    report = {
        "reproduced": reproduced,
        "tolerance_auc": cfg.reproduction_auc_tolerance,
        "delta_auc": round(delta_auc, 6),
        "test_cutoffs": list(cfg.test_cutoffs),
        "test_rows": len(test),
        "champion": {"bundle_name": champion.bundle_name, "metrics": incumbent,
                     "recorded_metrics": champion.bundle.get("metrics", {})},
        "candidate": {"bundle_name": summary["bundle_name"], "candidate_path": summary["candidate_path"],
                      "calibration_method": summary["calibration_method"],
                      "best_iteration": summary["best_iteration"], "metrics": candidate},
        "data_source": data.source,
        "data_fingerprint": data.fingerprint(),
        "git_sha": summary["git_sha"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"reproduction_report_{cfg.target}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{'metric':<16}{'champion':>12}{'candidate':>12}{'delta':>10}")
    for m in COMPARED_METRICS:
        print(f"{m:<16}{incumbent[m]:>12.4f}{candidate[m]:>12.4f}{candidate[m] - incumbent[m]:>+10.4f}")
    verdict = "REPRODUCED" if reproduced else "NOT REPRODUCED"
    print(f"\n{verdict}: |dAUC| = {abs(delta_auc):.4f} (tolerance {cfg.reproduction_auc_tolerance})")
    print(f"Report: {report_path}")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
