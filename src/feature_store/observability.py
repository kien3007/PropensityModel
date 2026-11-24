"""
Feature store observability with Evidently.

Every stored snapshot can be profiled against the snapshot before it:
  - per-column drift   Wasserstein distance (numeric) / Jensen-Shannon (categorical), both
                       normalised to [0, 1]; a column drifts when its score >= DRIFT_THRESHOLD.
                       Fixed methods (not Evidently's size-dependent auto choice) keep scores
                       comparable between the real store (2,500 rows) and small test stores.
  - out of range       share of values outside the [min, max] seen in the previous snapshot
  - column health      constant / almost-constant columns
  - summary statistics mean, std, min, max, quartiles (numeric); category shares (categorical)

Outputs, under <report_dir> (default OUTPUT_DIR/reports/feature_store):
  snapshot_<day>.html   the interactive Evidently report (distributions overlaid, per column)
  snapshot_<day>.json   the compact summary, also used for trends across snapshots

This is observation, not a gate: the blocking checks live in src/features/quality.py.
Evidently is imported lazily so the serving image does not need it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import settings
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("FeatureStoreObservability")

NUMERIC_DRIFT_METHOD = "wasserstein"
CATEGORICAL_DRIFT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1


def default_report_dir() -> Path:
    return Path(settings.OUTPUT_DIR) / "reports" / "feature_store"


def _metric_type(metric: Dict[str, Any]) -> str:
    return metric["config"]["type"].rsplit(":", 1)[-1]


def _summarise(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turns Evidently's metric list into a compact, JSON-friendly summary."""
    drift: Dict[str, Dict[str, Any]] = {}
    out_of_range: Dict[str, Dict[str, float]] = {}
    stats: Dict[str, Dict[str, float]] = {f: {} for f in NUMERIC_FEATURES}
    categories: Dict[str, Dict[str, float]] = {}
    health: Dict[str, float] = {}
    for m in metrics:
        kind, cfg, value = _metric_type(m), m["config"], m["value"]
        column = cfg.get("column")
        if kind == "ValueDrift":
            score = float(value)
            drift[column] = {"method": cfg["method"], "score": round(score, 6),
                             "drifted": bool(score >= cfg["threshold"])}
        elif kind == "OutRangeValueCount":
            out_of_range[column] = {"count": int(value["count"]), "share": round(float(value["share"]), 6),
                                    "reference_min": cfg["left"], "reference_max": cfg["right"]}
        elif kind in ("MeanValue", "StdValue", "MinValue", "MaxValue") and column in stats:
            stats[column][kind[:-5].lower()] = round(float(value), 6)
        elif kind == "QuantileValue" and column in stats:
            stats[column][f"p{int(round(cfg['quantile'] * 100))}"] = round(float(value), 6)
        elif kind == "UniqueValueCount" and column in CATEGORICAL_FEATURES:
            categories[column] = {str(k): round(float(v), 6) for k, v in value["shares"].items()}
        elif kind in ("ConstantColumnsCount", "AlmostConstantColumnsCount", "RowCount"):
            health[kind] = float(value)

    drifted = sorted(c for c, d in drift.items() if d["drifted"])
    return {
        "rows": int(health.get("RowCount", 0)),
        "constant_columns": int(health.get("ConstantColumnsCount", 0)),
        "almost_constant_columns": int(health.get("AlmostConstantColumnsCount", 0)),
        "drift": {
            "methods": {"numeric": NUMERIC_DRIFT_METHOD, "categorical": CATEGORICAL_DRIFT_METHOD},
            "threshold": DRIFT_THRESHOLD,
            "drifted_columns": drifted,
            "drifted_count": len(drifted),
            "drifted_share": round(len(drifted) / len(drift), 6) if drift else None,
            "columns": drift,
        } if drift else None,
        "out_of_range": out_of_range or None,
        "max_out_of_range_share": max((v["share"] for v in out_of_range.values()), default=None),
        "numeric_stats": stats,
        "category_shares": categories,
    }


def profile_snapshot(
    current: pd.DataFrame,
    cutoff_day: int,
    reference: Optional[pd.DataFrame] = None,
    reference_day: Optional[int] = None,
    report_dir: Optional[Path | str] = None,
    save_html: bool = True,
) -> Dict[str, Any]:
    """
    Profiles one snapshot (against the previous one when given) and writes the HTML report and
    the JSON summary. Returns the summary.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import AlmostConstantColumnsCount, OutRangeValueCount
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    definition = DataDefinition(numerical_columns=list(NUMERIC_FEATURES),
                                categorical_columns=list(CATEGORICAL_FEATURES))

    def to_dataset(df: pd.DataFrame):
        frame = df[MODEL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype(str)
        for col in NUMERIC_FEATURES:
            frame[col] = frame[col].astype("float64")
        return Dataset.from_pandas(frame, data_definition=definition)

    metrics: list = [DataSummaryPreset(), AlmostConstantColumnsCount()]
    if reference is not None:
        metrics.append(DataDriftPreset(num_method=NUMERIC_DRIFT_METHOD, cat_method=CATEGORICAL_DRIFT_METHOD,
                                       threshold=DRIFT_THRESHOLD))
        metrics.extend(
            OutRangeValueCount(column=f, left=float(reference[f].min()), right=float(reference[f].max()))
            for f in NUMERIC_FEATURES
        )
    result = Report(metrics).run(to_dataset(current), to_dataset(reference) if reference is not None else None)

    out_dir = Path(report_dir or default_report_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cutoff_day": int(cutoff_day),
               "reference_day": int(reference_day) if reference_day is not None else None,
               **_summarise(result.dict()["metrics"])}
    if save_html:
        html_path = out_dir / f"snapshot_{int(cutoff_day)}.html"
        result.save_html(str(html_path))
        summary["html_report"] = str(html_path)
    (out_dir / f"snapshot_{int(cutoff_day)}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    drift = summary["drift"]
    logger.info(
        f"Feature snapshot DAY={cutoff_day} profiled"
        + (f" vs DAY {reference_day}: {drift['drifted_count']} drifted column(s) {drift['drifted_columns']}, "
           f"max out-of-range share {summary['max_out_of_range_share']:.3f}" if drift else " (no previous snapshot)")
    )
    return summary


def profile_stored_snapshot(client, cutoff_day: int, report_dir: Optional[Path | str] = None,
                            save_html: bool = True) -> Dict[str, Any]:
    """Profiles a snapshot already in the store against the stored snapshot before it."""
    previous_day = client.previous_snapshot_day(cutoff_day)
    return profile_snapshot(
        client.read_snapshot(cutoff_day), cutoff_day,
        reference=client.read_snapshot(previous_day) if previous_day is not None else None,
        reference_day=previous_day, report_dir=report_dir, save_html=save_html,
    )


def load_profiles(report_dir: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """All stored snapshot summaries, oldest first (for trends across snapshots)."""
    out_dir = Path(report_dir or default_report_dir())
    profiles = [json.loads(p.read_text(encoding="utf-8")) for p in out_dir.glob("snapshot_*.json")]
    return sorted(profiles, key=lambda p: p["cutoff_day"])


def trend(feature: str, report_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """One row per profiled snapshot: summary statistics and drift score of one feature."""
    rows = []
    for p in load_profiles(report_dir):
        row: Dict[str, Any] = {"cutoff_day": p["cutoff_day"]}
        if feature in NUMERIC_FEATURES:
            row.update(p["numeric_stats"].get(feature, {}))
            oor = (p.get("out_of_range") or {}).get(feature)
            row["out_of_range_share"] = oor["share"] if oor else None
        else:
            row.update(p["category_shares"].get(feature, {}))
        col = ((p.get("drift") or {}).get("columns") or {}).get(feature)
        row["drift_score"] = col["score"] if col else None
        rows.append(row)
    return pd.DataFrame(rows)
