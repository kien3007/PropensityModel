"""
Enterprise Model & Data Drift Monitoring Module.
Implements Population Stability Index (PSI), two-sample Kolmogorov-Smirnov (KS) test,
and categorical frequency shift detection across 28 multi-domain features and prediction scores.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import settings
from src.features.contract import MODEL_FEATURES

logger = logging.getLogger("DriftDetector")

# Production thresholds
PSI_STABLE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05


class DataDriftDetector:
    """
    Statistical engine for detecting Covariate Shift, Prediction Drift, and Schema Stability.
    """

    def __init__(self, output_dir: Optional[Path | str] = None) -> None:
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_numerical_psi(
        reference: np.ndarray,
        current: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Calculates Population Stability Index (PSI) for continuous numerical distributions.
        PSI = sum((Q_i - P_i) * ln(Q_i / P_i))
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0

        # Edge case: constant column
        if np.all(ref_clean == ref_clean[0]) and np.all(cur_clean == cur_clean[0]):
            return 0.0 if ref_clean[0] == cur_clean[0] else 1.0

        # Define quantile bins on reference dataset
        quantiles = np.linspace(0, 100, num_buckets + 1)
        try:
            bins = np.percentile(ref_clean, quantiles)
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.linspace(min(ref_clean.min(), cur_clean.min()), max(ref_clean.max(), cur_clean.max()), num_buckets + 1)
                bins = np.unique(bins)
        except Exception:
            bins = np.histogram_bin_edges(ref_clean, bins=num_buckets)

        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        # Compute bucket proportions
        ref_counts, _ = np.histogram(ref_clean, bins=bins)
        cur_counts, _ = np.histogram(cur_clean, bins=bins)

        ref_pct = (ref_counts + epsilon) / (len(ref_clean) + epsilon * len(ref_counts))
        cur_pct = (cur_counts + epsilon) / (len(cur_clean) + epsilon * len(cur_counts))

        # Calculate PSI
        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(np.round(psi_value, 4))

    @staticmethod
    def calculate_ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """
        Two-sample Kolmogorov-Smirnov test to check whether two underlying distributions differ.
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return {"statistic": 0.0, "p_value": 1.0}

        res = stats.ks_2samp(ref_clean, cur_clean)
        return {
            "statistic": float(np.round(res.statistic, 4)),
            "p_value": float(np.round(res.pvalue, 6)),
        }

    @staticmethod
    def calculate_categorical_psi(
        reference: pd.Series,
        current: pd.Series,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Calculates Population Stability Index across discrete category bins.
        """
        all_categories = list(set(reference.dropna().unique()).union(set(current.dropna().unique())))
        if not all_categories:
            return 0.0

        ref_counts = reference.value_counts(normalize=True).to_dict()
        cur_counts = current.value_counts(normalize=True).to_dict()

        total_psi = 0.0
        for cat in all_categories:
            p = ref_counts.get(cat, 0.0) + epsilon
            q = cur_counts.get(cat, 0.0) + epsilon
            total_psi += (q - p) * np.log(q / p)

        return float(np.round(total_psi, 4))

    def evaluate_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates data drift for all features between reference (training/baseline) and current batch.
        """
        features = features or MODEL_FEATURES
        categorical_features = categorical_features or [
            "classification_1",
            "classification_2",
            "classification_3",
            "HOMEOWNER_DESC",
            "classification_4",
            "classification_5",
            "KID_CATEGORY_DESC",
        ]

        feature_metrics: Dict[str, Any] = {}
        drifted_count = 0
        warning_count = 0

        for feat in features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue

            if feat in categorical_features:
                psi = self.calculate_categorical_psi(reference_df[feat], current_df[feat])
                ks_res = {"statistic": None, "p_value": None}
                is_drifted = psi >= PSI_CRITICAL_THRESHOLD
                is_warning = PSI_STABLE_THRESHOLD <= psi < PSI_CRITICAL_THRESHOLD
            else:
                ref_arr = pd.to_numeric(reference_df[feat], errors="coerce").to_numpy(dtype=float)
                cur_arr = pd.to_numeric(current_df[feat], errors="coerce").to_numpy(dtype=float)

                psi = self.calculate_numerical_psi(ref_arr, cur_arr)
                ks_res = self.calculate_ks_test(ref_arr, cur_arr)

                is_drifted = (psi >= PSI_CRITICAL_THRESHOLD) or (ks_res["p_value"] < KS_PVALUE_THRESHOLD and psi >= 0.15)
                is_warning = (PSI_STABLE_THRESHOLD <= psi < PSI_CRITICAL_THRESHOLD)

            status = "SIGNIFICANT_DRIFT" if is_drifted else ("MODERATE_DRIFT" if is_warning else "NO_DRIFT")
            if is_drifted:
                drifted_count += 1
            elif is_warning:
                warning_count += 1

            feature_metrics[feat] = {
                "type": "categorical" if feat in categorical_features else "numerical",
                "psi": psi,
                "ks_statistic": ks_res["statistic"],
                "ks_pvalue": ks_res["p_value"],
                "status": status,
            }

        drift_ratio = drifted_count / len(features) if features else 0.0

        return {
            "total_features_evaluated": len(feature_metrics),
            "drifted_features_count": drifted_count,
            "warning_features_count": warning_count,
            "drift_ratio": round(drift_ratio, 4),
            "feature_details": feature_metrics,
        }

    def evaluate_prediction_drift(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates drift on output calibrated purchase probabilities P(Y=1).
        """
        psi = self.calculate_numerical_psi(reference_scores, current_scores)
        ks = self.calculate_ks_test(reference_scores, current_scores)

        ref_mean = float(np.mean(reference_scores))
        cur_mean = float(np.mean(current_scores))
        mean_abs_shift = float(abs(cur_mean - ref_mean))

        status = "SIGNIFICANT_DRIFT" if psi >= PSI_CRITICAL_THRESHOLD else (
            "MODERATE_DRIFT" if psi >= PSI_STABLE_THRESHOLD else "NO_DRIFT"
        )

        return {
            "psi": psi,
            "ks_statistic": ks["statistic"],
            "ks_pvalue": ks["p_value"],
            "reference_mean_probability": round(ref_mean, 4),
            "current_mean_probability": round(cur_mean, 4),
            "mean_shift": round(mean_abs_shift, 4),
            "status": status,
        }

    def generate_full_drift_audit(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_scores: Optional[np.ndarray] = None,
        current_scores: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
        target: str = "purchase",
        reference_cutoffs: Optional[List[int]] = None,
        current_cutoff: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates full production drift report including feature drift, prediction drift,
        and recommendation on whether model retraining is urgently required.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or f"DRIFT_{timestamp}"

        # 1. Feature Drift
        feat_report = self.evaluate_feature_drift(reference_df, current_df)

        # 2. Prediction Drift
        pred_report = None
        if reference_scores is not None and current_scores is not None:
            pred_report = self.evaluate_prediction_drift(reference_scores, current_scores)

        # 3. Overall Verdict
        crit_drift = (feat_report["drift_ratio"] >= 0.20) or (
            pred_report is not None and pred_report["status"] == "SIGNIFICANT_DRIFT"
        )
        mod_drift = (feat_report["warning_features_count"] >= 3) or (
            pred_report is not None and pred_report["status"] == "MODERATE_DRIFT"
        )

        overall_status = "CRITICAL_DRIFT" if crit_drift else ("WARNING" if mod_drift else "STABLE")
        action_recommended = "TRIGGER_RETRAINING" if crit_drift else (
            "MONITOR_CLOSELY" if mod_drift else "NO_ACTION"
        )

        audit_report: Dict[str, Any] = {
            "run_id": run_id,
            "target": target,
            "reference_cutoffs": reference_cutoffs,
            "current_cutoff": current_cutoff,
            "evaluated_at": datetime.datetime.now().isoformat(),
            "overall_status": overall_status,
            "action_recommended": action_recommended,
            "requires_retraining": crit_drift,
            "feature_drift_summary": {
                "total_features": feat_report["total_features_evaluated"],
                "drifted_count": feat_report["drifted_features_count"],
                "warning_count": feat_report["warning_features_count"],
                "drift_ratio": feat_report["drift_ratio"],
            },
            "prediction_drift": pred_report,
            "feature_details": feat_report["feature_details"],
        }

        # 4. Save to JSON report
        report_file = self.output_dir / f"drift_report_{run_id}.json"
        with open(report_file, "w") as f:
            json.dump(audit_report, f, indent=2)

        latest_file = self.output_dir / f"drift_report_{target}_latest_full.json"
        with open(latest_file, "w") as f:
            json.dump(audit_report, f, indent=2)

        logger.info(
            f"Drift Audit '{run_id}' completed. Status: {overall_status} | "
            f"Action: {action_recommended} | Report: {report_file}"
        )

        return audit_report
