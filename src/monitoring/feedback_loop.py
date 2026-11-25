"""
Ground-Truth Feedback Loop.

Once a prediction's horizon has elapsed, joins the champion's predictions at a cutoff with
what actually happened and compares the realized quality with the quality the champion was
accepted with (its out-of-time test metrics):

- labels use the *training* definition (TargetSpec on trip-level baskets via the shared
  DataLoader / MultiDomainFeatureBuilder), never a separate CSV query;
- the horizon is the champion's own horizon;
- predictions come from Gold `fct_weekly_batch_scores` when the batch ran at that cutoff,
  otherwise the champion re-scores the cutoff ("replay", e.g. for backfills);
- degraded = expected AUC - realized AUC > settings.REALIZED_AUC_DROP_ALERT.

Results go to Gold `fct_model_attribution`, a JSON report, and MonitoringState (-> Prometheus).
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config.model_config import TRAINING_CONFIGS
from src.config.settings import settings
from src.data.loader import DataLoader, RawDataBundle
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.features.targets import get_target
from src.lakehouse.engine import LakehouseEngine
from src.models.predictor import PropensityPredictor
from src.monitoring.state import MonitoringState
from src.training.evaluator import ModelEvaluator
from src.training.model_registry import registry_dir_for

logger = logging.getLogger("FeedbackLoop")

ATTRIBUTION_COLUMNS = {
    "eval_id": "VARCHAR",
    "target": "VARCHAR",
    "cutoff_day": "INTEGER",
    "horizon_days": "INTEGER",
    "evaluated_at": "TIMESTAMP",
    "model_bundle": "VARCHAR",
    "prediction_source": "VARCHAR",
    "sample_size": "INTEGER",
    "base_rate": "DOUBLE",
    "realized_roc_auc": "DOUBLE",
    "expected_roc_auc": "DOUBLE",
    "auc_drop": "DOUBLE",
    "realized_brier_score": "DOUBLE",
    "realized_log_loss": "DOUBLE",
    "realized_top_decile_lift": "DOUBLE",
    "decile_10_actual_rate": "DOUBLE",
    "attribution_status": "VARCHAR",
}


class GroundTruthFeedbackLoop:
    def __init__(
        self,
        lakehouse_engine: Optional[LakehouseEngine] = None,
        output_dir: Optional[Path | str] = None,
        data: Optional[RawDataBundle] = None,
        state: Optional[MonitoringState] = None,
    ) -> None:
        self.lakehouse = lakehouse_engine or LakehouseEngine()
        self.output_dir = Path(output_dir or (settings.OUTPUT_DIR / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data = data
        self._builder: Optional[MultiDomainFeatureBuilder] = None
        self.state = state or MonitoringState(report_dir=self.output_dir)
        self._ensure_attribution_table()

    @property
    def builder(self) -> MultiDomainFeatureBuilder:
        if self._builder is None:
            self._builder = MultiDomainFeatureBuilder(self._data or DataLoader.from_lakehouse(self.lakehouse))
        return self._builder

    def _ensure_attribution_table(self) -> None:
        con = self.lakehouse.get_connection()
        try:
            cols = ", ".join(f"{c} {t}" for c, t in ATTRIBUTION_COLUMNS.items())
            con.execute(f"CREATE TABLE IF NOT EXISTS fct_model_attribution ({cols});")
            for col, typ in ATTRIBUTION_COLUMNS.items():  # upgrade tables created by older versions
                con.execute(f"ALTER TABLE fct_model_attribution ADD COLUMN IF NOT EXISTS {col} {typ};")
        finally:
            con.close()

    # ------------------------------------------------------------ predictions
    def _stored_predictions(self, cutoff_day: int) -> pd.DataFrame:
        con = self.lakehouse.get_connection()
        try:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fct_weekly_batch_scores'"
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            return con.execute(
                "SELECT household_key, purchase_probability AS probability, decile "
                "FROM fct_weekly_batch_scores WHERE snapshot_day = ? "
                "QUALIFY ROW_NUMBER() OVER (PARTITION BY household_key ORDER BY scored_at DESC) = 1",
                [cutoff_day],
            ).df()
        finally:
            con.close()

    def _replay_predictions(self, predictor: PropensityPredictor, cutoff_day: int) -> pd.DataFrame:
        feats = self.builder.features_at(cutoff_day)
        feats = feats[get_target(predictor.target).eligible(feats)]
        scored = predictor.score_households(feats)
        prob_col = "lapse_probability" if predictor.target == "lapse" else "purchase_probability"
        return scored[["household_key", prob_col, "decile"]].rename(columns={prob_col: "probability"})

    # ------------------------------------------------------------- evaluation
    def evaluate_realized_performance(
        self,
        target: str = "purchase",
        cutoff_day: Optional[int] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        model_path: Optional[Path | str] = None,
        backfill: bool = True,
    ) -> Dict[str, Any]:
        start = datetime.datetime.now()
        cfg = TRAINING_CONFIGS[target]
        predictor = PropensityPredictor(
            model_path=model_path or (registry_dir_for(cfg) / "champion_model.joblib"), expected_target=target,
        )
        horizon = predictor.horizon_days
        last_day = self.builder.last_observed_day()
        cutoff = cutoff_day if cutoff_day is not None else last_day - horizon
        if cutoff + horizon > last_day:
            raise ValueError(f"Labels for cutoff {cutoff} mature on DAY {cutoff + horizon} > last observed DAY {last_day}")

        if predictions_df is not None:
            preds, source = predictions_df.rename(
                columns={"purchase_probability": "probability", "lapse_probability": "probability"}), "provided"
        else:
            preds, source = (self._stored_predictions(cutoff), "batch") if target == "purchase" else (pd.DataFrame(), "")
            if preds.empty:
                if not backfill:
                    raise ValueError(f"No stored {target} predictions for cutoff {cutoff} and backfill disabled")
                preds, source = self._replay_predictions(predictor, cutoff), "replay"

        labels = get_target(target).labels(self.builder, cutoff, horizon)
        joined = preds[["household_key", "probability", "decile"]].merge(
            labels[["household_key", "target"]], on="household_key", how="inner")
        if joined["target"].nunique() < 2:
            raise ValueError(f"Realized labels at cutoff {cutoff} are single-class; AUC undefined")

        m = ModelEvaluator.evaluate(joined["target"], joined["probability"])
        expected = float((predictor.bundle.get("metrics") or {}).get("roc_auc", np.nan))
        auc_drop = expected - m["roc_auc"]
        degraded = bool(np.isfinite(auc_drop) and auc_drop > settings.REALIZED_AUC_DROP_ALERT)
        top = joined[joined["decile"] == 10]
        decile_10_rate = float(top["target"].mean()) if len(top) else float("nan")

        deciles = (joined.groupby("decile")
                   .agg(count=("target", "size"), mean_predicted_prob=("probability", "mean"),
                        actual_rate=("target", "mean"))
                   .reset_index())
        deciles["lift"] = deciles["actual_rate"] / m["base_rate"] if m["base_rate"] > 0 else np.nan

        eval_id = f"FB_{target}_DAY_{cutoff}_{horizon}d_{start.strftime('%Y%m%d_%H%M%S_%f')}"
        result: Dict[str, Any] = {
            "eval_id": eval_id,
            "target": target,
            "cutoff_day": int(cutoff),
            "horizon_days": int(horizon),
            "evaluated_at": start.isoformat(timespec="seconds"),
            "model_bundle": predictor.bundle_name,
            "prediction_source": source,
            "sample_size": int(len(joined)),
            "base_rate": m["base_rate"],
            "realized_roc_auc": m["roc_auc"],
            "expected_roc_auc": round(expected, 6),
            "auc_drop": round(float(auc_drop), 6),
            "realized_brier_score": m["brier_score"],
            "realized_log_loss": m["log_loss"],
            "realized_ece": m["ece"],
            "realized_top_decile_lift": m["lift_at_10pct"],
            "decile_10_actual_rate": round(decile_10_rate, 4),
            "degraded": degraded,
            "attribution_status": "DEGRADED" if degraded else "HEALTHY",
            "decile_breakdown": deciles.round(4).to_dict("records"),
        }

        con = self.lakehouse.get_connection()
        try:
            cols = list(ATTRIBUTION_COLUMNS)
            row = [result[c] if c != "evaluated_at" else start for c in cols]
            con.execute(f"INSERT INTO fct_model_attribution ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", row)
        finally:
            con.close()

        report = self.output_dir / f"feedback_attribution_{target}_day_{cutoff}.json"
        report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        self.state.publish(f"feedback_{target}", result)
        log = logger.warning if degraded else logger.info
        log(f"Feedback {target}@{cutoff} ({source}): realized AUC {m['roc_auc']:.4f} vs accepted {expected:.4f} "
            f"(drop {auc_drop:+.4f}) -> {result['attribution_status']}")
        return result
