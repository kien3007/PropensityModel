"""
Target & Business-Value Analysis (Phase 3).

1. How predictable each target is: purchase base rate by horizon and by frequency segment.
2. For every target (purchase, lapse): out-of-time comparison of the LightGBM model,
   logistic regression, the rule-of-thumb heuristic and random selection at fixed weekly
   campaign capacities (top 5 / 10 / 20% of scored households per cutoff).
   - purchase value = spend the selected households make inside the horizon
   - lapse value    = 90-day spend (pre-cutoff, i.e. known at decision time) of correctly
                      flagged lapsers: the revenue run-rate a retention action could protect
3. Writes docs/analysis/TARGET_ANALYSIS.md and the raw capacity table as CSV.

Usage: python -m scripts.target_analysis [--data-source lakehouse|csv]
Nothing is promoted: models are trained with promote=False.
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config.model_config import TRAINING_CONFIGS
from src.config.settings import settings
from src.data.loader import DataLoader, RawDataBundle
from src.features.pit_extractor import MultiDomainFeatureBuilder
from src.pipelines.train_pipeline import run_training_pipeline

logger = logging.getLogger("TargetAnalysis")

CAPACITIES = (0.05, 0.10, 0.20)
SCORERS = {"LightGBM": "model_probability", "Logistic regression": "logistic_probability",
           "Heuristic rule": "heuristic_score"}
HEURISTIC_LABEL = {"purchase": "1 / (1 + days since last basket)", "lapse": "overdue ratio vs own cadence"}
DOCS_DIR = settings.BASE_DIR / "docs" / "analysis"


def predictability_tables(builder: MultiDomainFeatureBuilder) -> Dict[str, pd.DataFrame]:
    cutoffs = [200, 260, 320, 380, 440, 500, 560, 620, 680]
    last_day = builder.last_observed_day()
    horizon_rows = []
    for h in (3, 7, 14, 21, 28):
        rates = [builder.labels_at(c, h)["target"].mean() for c in cutoffs if c + h <= last_day]
        horizon_rows.append({"horizon_days": str(h), "mean_rate": np.mean(rates), "min_rate": min(rates),
                             "max_rate": max(rates), "max_lift_10": 1 / np.mean(rates)})

    frames = []
    for c in (500, 560, 620, 680):
        frames.append(builder.features_at(c).merge(builder.labels_at(c, 14), on=["household_key", "cutoff_day"]))
    stacked = pd.concat(frames)
    segment = pd.cut(stacked["baskets_count_90d"], [-1, 0, 3, 9, 19, np.inf], labels=["0", "1-3", "4-9", "10-19", "20+"])
    seg_table = (stacked.groupby(segment, observed=True)["target"].agg(["mean", "size"])
                 .rename(columns={"mean": "purchase_rate_14d", "size": "rows"}).reset_index())
    seg_table["share_of_rows"] = seg_table["rows"] / seg_table["rows"].sum()
    return {"horizon": pd.DataFrame(horizon_rows), "segment": seg_table}


def future_spend(builder: MultiDomainFeatureBuilder, cutoff: int, horizon: int) -> pd.Series:
    b = builder.baskets
    window = b[(b["DAY"] > cutoff) & (b["DAY"] <= cutoff + horizon)]
    return window.groupby("household_key")["basket_spend"].sum()


def capacity_table(preds: pd.DataFrame, builder: MultiDomainFeatureBuilder, target: str, horizon: int) -> pd.DataFrame:
    rows: List[dict] = []
    rng = np.random.default_rng(0)
    for cutoff, grp in preds.groupby("cutoff_day"):
        grp = grp.copy()
        if target == "purchase":
            grp["value"] = grp["household_key"].map(future_spend(builder, cutoff, horizon)).fillna(0.0)
        else:
            spend90 = grp["household_key"].map(builder.features_at(cutoff).set_index("household_key")["spend_90d"])
            grp["value"] = spend90.fillna(0.0) * grp["target"]
            # Expected spend at risk: rank by p(lapse) x 90-day spend instead of p(lapse) alone.
            grp["expected_value_score"] = grp["model_probability"] * spend90.fillna(0.0)
        grp["random_score"] = rng.uniform(size=len(grp))
        positives, total_value = grp["target"].sum(), grp["value"].sum()
        for cap in CAPACITIES:
            k = max(1, int(round(cap * len(grp))))
            scorers = {**SCORERS, "Random": "random_score"}
            if target == "lapse":
                scorers = {"LightGBM x 90d spend": "expected_value_score", **scorers}
            for name, col in scorers.items():
                top = grp.nlargest(k, col)
                rows.append({"target": target, "capacity": cap, "cutoff_day": cutoff, "scorer": name, "k": k,
                             "hits": int(top["target"].sum()), "positives": int(positives),
                             "value": float(top["value"].sum()), "total_value": float(total_value)})
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["target", "capacity", "scorer"], sort=False).sum(numeric_only=True).reset_index()
    agg["precision"] = agg["hits"] / agg["k"]
    agg["capture_rate"] = agg["hits"] / agg["positives"]
    agg["value_share"] = agg["value"] / agg["total_value"]
    random = agg[agg["scorer"] == "Random"].set_index("capacity")
    agg["value_vs_random"] = agg.apply(lambda r: r["value"] / random.loc[r["capacity"], "value"], axis=1)
    return agg.drop(columns=["cutoff_day"])


def winners(cap: pd.DataFrame) -> pd.DataFrame:
    """Best scorer per capacity by positives caught and by value captured."""
    rows = []
    for c, g in cap[cap["scorer"] != "Random"].groupby("capacity"):
        by_hits = g.loc[g["hits"].idxmax()]
        by_value = g.loc[g["value"].idxmax()]
        lgbm = g[g["scorer"] == "LightGBM"].iloc[0]
        lr = g[g["scorer"] == "Logistic regression"].iloc[0]
        rows.append({
            "capacity": f"top {c * 100:.0f}%",
            "most positives caught": f"{by_hits['scorer']} ({int(by_hits['hits'])})",
            "most value captured": f"{by_value['scorer']} ({by_value['value']:,.0f})",
            "LightGBM vs LogReg hits": f"{int(lgbm['hits'] - lr['hits']):+d}",
            "LightGBM vs LogReg value": f"{lgbm['value'] - lr['value']:+,.0f}",
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: Dict[str, str]) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt[c]) if c in floatfmt and isinstance(v, (int, float, np.number)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_report(data: RawDataBundle, out_dir: Path) -> Path:
    builder = MultiDomainFeatureBuilder(data)
    pred_tables = predictability_tables(builder)
    sections, capacity_frames = [], []
    for target, cfg in TRAINING_CONFIGS.items():
        summary = run_training_pipeline(config=cfg, data=data, output_dir=out_dir / target, promote=False)
        preds = pd.read_parquet(summary["test_predictions_path"])
        cap = capacity_table(preds, builder, target, cfg.horizon_days)
        capacity_frames.append(cap)

        metrics = pd.DataFrame([
            {"scorer": "LightGBM", **summary["test_metrics"]},
            {"scorer": "Logistic regression", **summary["baselines"]["logistic_regression"]},
            {"scorer": f"Heuristic ({HEURISTIC_LABEL[target]})", **summary["baselines"]["recency_rule"]},
        ])[["scorer", "roc_auc", "pr_auc", "lift_at_10pct", "brier_skill_score", "ece"]]
        view = cap[["capacity", "scorer", "k", "precision", "capture_rate", "value", "value_vs_random"]].copy()
        win = winners(cap)
        view["capacity"] = (view["capacity"] * 100).map(lambda x: f"top {x:.0f}%")
        value_label = "spend in horizon ($)" if target == "purchase" else "90d spend of caught lapsers ($)"
        view = view.rename(columns={"value": value_label, "k": "households"})
        sections.append(
            f"### Target `{target}` (horizon {cfg.horizon_days} days, test cutoffs {list(cfg.test_cutoffs)})\n\n"
            f"Test rows: {summary['rows']['test']:,}; base rate {summary['test_metrics']['base_rate']:.3f}; "
            f"max attainable lift@10% {summary['test_metrics']['max_attainable_lift_10']:.2f}x.\n\n"
            + md_table(metrics, {"roc_auc": ".4f", "pr_auc": ".4f", "lift_at_10pct": ".2f",
                                 "brier_skill_score": ".3f", "ece": ".4f"})
            + "\n\nCapacity scenarios (summed over test cutoffs):\n\n"
            + md_table(view, {"precision": ".3f", "capture_rate": ".3f", value_label: ",.0f",
                              "value_vs_random": ".2f"})
            + "\n\nWho wins at each capacity:\n\n" + md_table(win, {})
        )

    capacity = pd.concat(capacity_frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(out_dir / "capacity_scenarios.csv", index=False)

    horizon = pred_tables["horizon"]
    report = f"""# Target & Business-Value Analysis

> Generated by `python -m scripts.target_analysis` on {datetime.date.today().isoformat()} from {data.source}
> (data fingerprint `{data.fingerprint()}`). All model numbers are out-of-time test results; nothing was promoted.

## 1. How predictable is "will the household buy?"

Purchase base rate by horizon (mean over 9 cutoffs between DAY 200 and 680):

{md_table(horizon, {"mean_rate": ".3f", "min_rate": ".3f", "max_rate": ".3f", "max_lift_10": ".2f"})}

14-day purchase rate by shopping frequency in the previous 90 days (cutoffs 500-680):

{md_table(pred_tables["segment"], {"purchase_rate_14d": ".3f", "share_of_rows": ".3f"})}

At 14 days most households buy, and frequency alone almost decides the label. The ceiling
on top-decile lift is ~1.4x. A model can rank these households well (high AUC), but
reaching the top decile mostly reaches people who were going to shop anyway.

## 2. Model vs. baselines at fixed campaign capacity

{chr(10).join(chr(10) + s + chr(10) for s in sections)}
## 3. Reading the results

The tables above are regenerated from data; the notes below were written for data fingerprint
`385c047b9baa0a81` and should be re-checked if the "who wins" tables change.

- **purchase**: every model concentrates in-horizon spend 2-3.6x better than random, but the
  households selected are the heavy, regular shoppers who were going to buy anyway (precision
  ~1.0 in the top decile). Without a holdout this spend is not incremental, so it is not campaign
  value. Logistic regression matches LightGBM on hits and captures more spend.
- **lapse**: ranking matters here (base rate ~11%). Ranked by p(lapse), the model catches ~1.6-1.9x
  more lapsers than the overdue heuristic. The heuristic, however, protects more spend at small
  capacities, because it favours big spenders who are overdue. Ranking by **p(lapse) x 90-day
  spend** combines both: use it for value-driven retention lists and p(lapse) for reach.
- **LightGBM vs logistic regression**: the gap is small for both targets. The 28 features are
  mostly monotone frequency/recency signals, so a linear model captures most of the value. Keep
  logistic regression as a mandatory baseline, and ask LightGBM to justify its complexity.
- Value columns measure *association*, not causal uplift. The next step toward real business value
  is an uplift / holdout design (treated vs. control inside the targeted deciles).

Raw numbers: `data/outputs/analysis/capacity_scenarios.csv`.
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "TARGET_ANALYSIS.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-source", choices=["lakehouse", "csv"], default=None)
    parser.add_argument("--output-dir", type=Path, default=settings.OUTPUT_DIR / "analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = build_report(DataLoader.load(args.data_source), args.output_dir)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
