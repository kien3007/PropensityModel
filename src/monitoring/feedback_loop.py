"""
Ground-Truth Feedback Loop & Model Attribution Module.
Evaluates historical predictions against realized actual purchases occurring 7 to 14 days later.
Calculates Realized ROC-AUC, Brier Score, Log Loss, Decile Lift, and Calibration Curves.
Persists performance attribution to Lakehouse Gold tables and JSON reports.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.config.settings import settings
from src.lakehouse.engine import LakehouseEngine

logger = logging.getLogger("FeedbackLoop")


class GroundTruthFeedbackLoop:
    """
    Automated feedback loop joining historical predictions with realized post-prediction transactions.
    """

    def __init__(
        self,
        lakehouse_engine: Optional[LakehouseEngine] = None,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.lakehouse = lakehouse_engine or LakehouseEngine()
        self.output_dir = Path(output_dir or (settings.BASE_DIR / "output" / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_attribution_table()

    def _ensure_attribution_table(self) -> None:
        """Creates fct_model_attribution table in Gold Lakehouse if not exists."""
        con = self.lakehouse.get_connection()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS fct_model_attribution (
                    eval_id VARCHAR PRIMARY KEY,
                    cutoff_day INTEGER,
                    horizon_days INTEGER,
                    evaluated_at TIMESTAMP,
                    sample_size INTEGER,
                    base_rate DOUBLE,
                    realized_roc_auc DOUBLE,
                    realized_brier_score DOUBLE,
                    realized_log_loss DOUBLE,
                    realized_top_decile_lift DOUBLE,
                    decile_10_actual_rate DOUBLE,
                    attribution_status VARCHAR
                );
            """)
        finally:
            con.close()

    def evaluate_realized_performance(
        self,
        cutoff_day: int,
        horizon_days: int = 14,
        predictions_df: Optional[pd.DataFrame] = None,
        raw_tx_csv: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """
        Joins predictions at cutoff_day with actual transactions in (cutoff_day, cutoff_day + horizon_days].
        """
        start_time = datetime.datetime.now()
        eval_id = f"FB_DAY_{cutoff_day}_{horizon_days}d_{start_time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"=== Starting Ground-Truth Attribution: {eval_id} (Horizon: {cutoff_day} -> {cutoff_day + horizon_days}) ===")

        # 1. Fetch Historical Predictions
        if predictions_df is None:
            # Query from Lakehouse Gold fct_weekly_batch_scores or output parquet
            con = self.lakehouse.get_connection()
            try:
                scores_query = f"""
                    SELECT household_key, purchase_probability, decile, urgency_tier
                    FROM fct_weekly_batch_scores
                    WHERE snapshot_day = {cutoff_day}
                """
                preds = con.execute(scores_query).df()
            finally:
                con.close()

            if len(preds) == 0:
                # Fallback to output parquet
                pq_file = settings.OUTPUT_DIR / f"household_scores_day_{cutoff_day}.parquet"
                if pq_file.exists():
                    preds = pd.read_parquet(pq_file)
                else:
                    raise ValueError(f"No historical predictions found for cutoff day {cutoff_day}")
        else:
            preds = predictions_df.copy()

        # 2. Extract Realized Transactions in Prospective Window
        target_start = cutoff_day + 1
        target_end = cutoff_day + horizon_days

        tx_path = Path(raw_tx_csv or (settings.DATA_RAW_DIR / "transaction_data.csv"))
        if not tx_path.exists():
            raise FileNotFoundError(f"Transactions dataset not found at {tx_path}")

        con = duckdb.connect()
        try:
            tx_escaped = str(tx_path).replace("\\", "/")
            actual_df = con.execute(f"""
                SELECT DISTINCT household_key::BIGINT AS household_key, 1 AS actual_purchase
                FROM read_csv_auto('{tx_escaped}')
                WHERE DAY >= {target_start} AND DAY <= {target_end} AND SALES_VALUE >= 0;
            """).df()
        finally:
            con.close()

        # 3. Join Predictions with Realized Ground-Truth
        joined = preds[["household_key", "purchase_probability", "decile"]].merge(
            actual_df, on="household_key", how="left"
        )
        joined["actual_purchase"] = joined["actual_purchase"].fillna(0).astype(int)

        y_true = joined["actual_purchase"].to_numpy()
        y_prob = joined["purchase_probability"].to_numpy()

        # 4. Compute Realized Performance Metrics
        base_rate = float(np.mean(y_true))

        # Check for binary variety
        if len(np.unique(y_true)) > 1:
            realized_auc = float(roc_auc_score(y_true, y_prob))
            realized_brier = float(brier_score_loss(y_true, y_prob))
            realized_logloss = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
        else:
            realized_auc = 0.50
            realized_brier = float(brier_score_loss(y_true, y_prob))
            realized_logloss = 0.0

        # Decile 10 analysis
        top_decile = joined[joined["decile"] == 10]
        decile_10_rate = float(top_decile["actual_purchase"].mean()) if len(top_decile) > 0 else 0.0
        realized_lift = float(decile_10_rate / base_rate) if base_rate > 0 else 1.0

        # Decile-by-decile conversion table
        decile_summary = []
        for d in range(1, 11):
            sub = joined[joined["decile"] == d]
            cnt = len(sub)
            conv_rate = float(sub["actual_purchase"].mean()) if cnt > 0 else 0.0
            mean_pred = float(sub["purchase_probability"].mean()) if cnt > 0 else 0.0
            decile_summary.append({
                "decile": d,
                "count": cnt,
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_conversion_rate": round(conv_rate, 4),
                "lift": round(conv_rate / base_rate, 2) if base_rate > 0 else 1.0,
            })

        # Quality status
        if realized_auc >= 0.80 and realized_brier <= 0.16:
            attribution_status = "EXCELLENT"
        elif realized_auc >= 0.70:
            attribution_status = "HEALTHY"
        elif realized_auc >= 0.60:
            attribution_status = "DEGRADED"
        else:
            attribution_status = "CRITICAL_DRIFT"

        result: Dict[str, Any] = {
            "eval_id": eval_id,
            "cutoff_day": cutoff_day,
            "horizon_days": horizon_days,
            "evaluated_at": datetime.datetime.now().isoformat(),
            "sample_size": len(joined),
            "base_rate": round(base_rate, 4),
            "realized_roc_auc": round(realized_auc, 4),
            "realized_brier_score": round(realized_brier, 4),
            "realized_log_loss": round(realized_logloss, 4),
            "realized_top_decile_lift": round(realized_lift, 2),
            "decile_10_actual_conversion": round(decile_10_rate, 4),
            "attribution_status": attribution_status,
            "decile_breakdown": decile_summary,
            "duration_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }

        # 5. Persist to Lakehouse Gold
        con = self.lakehouse.get_connection()
        try:
            con.execute("""
                INSERT OR REPLACE INTO fct_model_attribution VALUES (
                    ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?
                );
            """, [
                eval_id,
                cutoff_day,
                horizon_days,
                len(joined),
                base_rate,
                realized_auc,
                realized_brier,
                realized_logloss,
                realized_lift,
                decile_10_rate,
                attribution_status,
            ])
        finally:
            con.close()

        # 6. Save JSON Report
        report_file = self.output_dir / f"feedback_attribution_day_{cutoff_day}.json"
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"Ground-Truth Attribution complete: Realized AUC={realized_auc:.4f}, "
            f"Brier={realized_brier:.4f}, Top Decile Lift={realized_lift:.2f}x ({attribution_status})"
        )

        return result
