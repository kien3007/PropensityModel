"""Build the management-ready and technically defensible CRISP-DM V2 notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    cells = [
        markdown(
            "# Customer Purchase Propensity V2\n"
            "## A CRISP-DM case study for weekly household prioritization\n\n"
            "**Decision:** Khi capacity có hạn, hộ nào nên được ưu tiên vì có xác suất "
            "phát sinh ít nhất một basket trong 7 ngày tới cao nhất?\n\n"
            "Notebook này là evidence trail: từ business framing, data audit, target "
            "validation, point-in-time features, rolling backtest đến deployment."
        ),
        markdown(
            "### Executive result\n"
            "Temporal holdout: **PR-AUC 0.8506**, **ROC-AUC 0.8303**, **Brier 0.1679**, "
            "**lift@10% 1.819x**. Kết luận: go cho controlled prioritization pilot; "
            "propensity không chứng minh incremental effect của coupon."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, "
            "precision_recall_curve, roc_curve)\n\n"
            "pd.set_option('display.max_columns', 100)\n"
            "ROOT = Path('..') if Path.cwd().name == 'notebooks' else Path('.')\n"
            "EDA = ROOT / 'artifacts' / 'propensity_v2' / 'data_understanding'\n"
            "MODEL = ROOT / 'artifacts' / 'propensity_v2' / 'model'\n"
            "metrics = json.loads((MODEL / 'metrics_v2.json').read_text(encoding='utf-8'))\n"
            "audit = json.loads((EDA / 'data_audit.json').read_text(encoding='utf-8'))\n"
            "TARGET = 'target_purchase_next_7d'\n"
            "metrics['selected_model'], metrics['test_metrics']"
        ),
        markdown(
            "# Phase 1 – Business Understanding\n\n"
            "Prediction unit là `(household_key, snapshot_day)`. Output gồm calibrated "
            "probability, rank và percentile. Primary use case là weekly prioritization, "
            "không phải tự động phát coupon.\n\n"
            "Success gates: PR-AUC ≥ 0.83, lift@10% ≥ 1.70, Brier ≤ 0.18, có temporal "
            "holdout và không leakage."
        ),
        markdown(
            "## Why a 7-day horizon?\n"
            "Horizon phải vừa đủ actionable và không biến hầu hết household thành positive."
        ),
        code(
            "horizons = pd.read_csv(EDA / 'target_horizon_summary.csv')\n"
            "horizons.style.format({'mean_target_rate': '{:.2%}', "
            "'min_target_rate': '{:.2%}', 'max_target_rate': '{:.2%}'})"
        ),
        code(
            "ax = horizons.plot.bar(x='horizon_days', y='mean_target_rate', legend=False, "
            "color=['#2F5597', '#70AD47', '#ED7D31'], title='Target rate by prediction horizon')\n"
            "ax.set_ylabel('Positive rate'); ax.set_xlabel('Horizon (days)')\n"
            "ax.set_ylim(0, 1); plt.tight_layout()"
        ),
        markdown(
            "**Decision:** giữ 7 ngày. Target rate 53.05% cân bằng hơn 14 ngày (68.26%) "
            "và 30 ngày (80.96%), đồng thời khớp weekly operating cadence."
        ),
        markdown("# Phase 2 – Data Understanding"),
        markdown("## Data landscape"),
        code(
            "inventory = pd.read_csv(EDA / 'dataset_inventory.csv')\n"
            "inventory[['file', 'size_mb', 'rows', 'columns', 'duplicate_rows', 'missing_cells']]"
        ),
        markdown(
            "Tám bảng gồm transaction, product, demographic, campaign, coupon/redemption "
            "và 36.8 triệu dòng product-store-week media causal."
        ),
        markdown("## Transaction integrity"),
        code(
            "quality_keys = ['rows', 'households', 'baskets', 'products', 'stores', "
            "'day_min', 'day_max', 'exact_duplicate_rows', 'multi_household_baskets', "
            "'multi_day_baskets', 'invalid_trans_time_rows']\n"
            "pd.Series({key: audit['transaction_quality'][key] for key in quality_keys}, "
            "name='value').to_frame()"
        ),
        markdown("## A data issue that changed the model"),
        code(
            "pd.Series(audit['transaction_quality']['quantity_distribution'], "
            "name='QUANTITY').to_frame()"
        ),
        markdown(
            "`QUANTITY` đạt 89,638 và chủ yếu bị scale ở KIOSK-GAS/MISC SALES. V2 loại "
            "feature này và dùng basket count, line items cùng diversity để tránh so sánh "
            "những đơn vị không tương thích."
        ),
        markdown("## Product and department mix"),
        code(
            "department = pd.read_csv(EDA / 'department_summary.csv').head(12)\n"
            "department[['DEPARTMENT', 'line_items', 'households', 'sales', 'sales_share']]"
        ),
        code(
            "plot_data = department.sort_values('sales_share')\n"
            "ax = plot_data.plot.barh(x='DEPARTMENT', y='sales_share', legend=False, "
            "title='Sales share by leading departments', color='#4472C4')\n"
            "ax.set_xlabel('Sales share'); plt.tight_layout()"
        ),
        markdown("## Temporal behavior"),
        code(
            "weekly = pd.read_csv(EDA / 'weekly_summary.csv')\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "weekly.plot(x='WEEK_NO', y='households', ax=axes[0], legend=False, "
            "title='Active households by week')\n"
            "weekly.plot(x='WEEK_NO', y='sales', ax=axes[1], legend=False, "
            "title='Observed sales by week', color='#70AD47')\n"
            "plt.tight_layout()"
        ),
        markdown("## Household heterogeneity"),
        code(
            "households = pd.read_csv(EDA / 'household_summary.csv')\n"
            "households[['active_days', 'baskets', 'spend', 'mean_gap_days', "
            "'avg_basket_value']].describe(percentiles=[.01, .25, .5, .75, .99]).T"
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
            "households['active_days'].clip("
            "upper=households['active_days'].quantile(.99)).hist("
            "bins=30, ax=axes[0]); axes[0].set_title('Active days (winsorized p99)')\n"
            "households['mean_gap_days'].clip("
            "upper=households['mean_gap_days'].quantile(.99)).hist("
            "bins=30, ax=axes[1]); axes[1].set_title('Mean purchase gap')\n"
            "households['avg_basket_value'].clip("
            "upper=households['avg_basket_value'].quantile(.99)).hist("
            "bins=30, ax=axes[2]); axes[2].set_title('Average basket value')\n"
            "plt.tight_layout()"
        ),
        markdown("## Join coverage and promotion data"),
        code(
            "pd.Series(audit['relational_integrity_and_coverage'], name='value').to_frame()"
        ),
        markdown(
            "Product join đạt 100%; campaign household coverage 63.36%; demographic "
            "coverage 32.04%; redemption đều khớp campaign window. Thiếu demographic "
            "được giữ là `Unknown`, không loại khỏi cohort."
        ),
        markdown("# Phase 3 – Data Preparation"),
        markdown(
            "## Point-in-time contract\n\n"
            "- Feature window: `(snapshot-window, snapshot]`.\n"
            "- Label window: `(snapshot, snapshot+7]`.\n"
            "- Household chỉ eligible sau lần mua đầu tiên.\n"
            "- Snapshot cách 7 ngày; embargo một horizon giữa partitions.\n"
            "- Không dùng future campaign/redemption outcome."
        ),
        code(
            "pd.DataFrame({key: value for key, value in metrics['split_manifest'].items() "
            "if isinstance(value, dict)}).T"
        ),
        markdown("## Feature architecture"),
        code(
            "modeling_table = pd.read_parquet(MODEL / 'modeling_table_v2.parquet')\n"
            "{'rows': len(modeling_table), 'columns': len(modeling_table.columns), "
            "'duplicate_keys': int(modeling_table.duplicated("
            "['household_key', 'snapshot_day']).sum()), "
            "'missing_cells': int(modeling_table.isna().sum().sum()), "
            "'target_rate': modeling_table[TARGET].mean()}"
        ),
        markdown(
            "170 features: cadence, basket, monetary, promotion, category, brand, "
            "daypart, assortment/store diversity, lifecycle và demographics, trên "
            "các cửa sổ 7/30/90/180 ngày."
        ),
        markdown("# Phase 4 – Modeling"),
        markdown(
            "Model family được chọn bằng ba expanding-window rolling-origin folds, "
            "không chọn theo test holdout."
        ),
        code(
            "selection = pd.read_csv(MODEL / 'model_selection_summary.csv')\n"
            "selection.style.format({'mean_pr_auc': '{:.4f}', 'std_pr_auc': '{:.4f}', "
            "'mean_roc_auc': '{:.4f}', 'mean_brier_score': '{:.4f}', "
            "'mean_lift_at_10pct': '{:.3f}'})"
        ),
        code(
            "backtest = pd.read_csv(MODEL / 'rolling_backtest_metrics.csv')\n"
            "pivot = backtest.pivot(index='fold', columns='model', values='pr_auc')\n"
            "ax = pivot.plot(marker='o', figsize=(10, 4), title='Rolling-origin PR-AUC')\n"
            "ax.set_ylabel('PR-AUC'); ax.grid(alpha=.3); plt.tight_layout()"
        ),
        markdown(
            "Regularized HGB thắng với mean PR-AUC 0.85885. Full logistic chỉ thấp hơn "
            "khoảng 0.0022 và được giữ làm challenger để cân bằng explainability/operations."
        ),
        markdown("# Phase 5 – Evaluation"),
        markdown("## Holdout metrics and uncertainty"),
        code("pd.Series(metrics['test_metrics'], name='test').to_frame()"),
        code(
            "intervals = pd.read_csv(MODEL / 'bootstrap_confidence_intervals.csv')\n"
            "intervals"
        ),
        markdown("## Calibration"),
        code(
            "calibration = pd.read_csv(MODEL / 'test_calibration.csv')\n"
            "ax = calibration.plot(x='predicted_probability', y='observed_rate', "
            "marker='o', "
            "xlim=(0,1), ylim=(0,1), grid=True, legend=False, "
            "title='Calibration by score decile')\n"
            "ax.plot([0,1], [0,1], '--', color='gray'); plt.tight_layout()"
        ),
        markdown("## ROC and precision-recall curves"),
        code(
            "predictions = pd.read_csv(MODEL / 'test_predictions.csv')\n"
            "y = predictions[TARGET].to_numpy(); "
            "p = predictions['purchase_probability'].to_numpy()\n"
            "fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            "axes[0].plot(fpr, tpr); axes[0].plot([0,1],[0,1],'--',color='gray'); "
            "axes[0].set(title='ROC curve', xlabel='FPR', ylabel='TPR')\n"
            "axes[1].plot(recall, precision); axes[1].axhline(y.mean(),ls='--',color='gray'); "
            "axes[1].set(title='Precision-recall curve', xlabel='Recall', ylabel='Precision')\n"
            "plt.tight_layout()"
        ),
        markdown("## Threshold view (diagnostic, not the default business rule)"),
        code(
            "threshold = metrics['test_metrics']['threshold']\n"
            "matrix = confusion_matrix(y, p >= threshold)\n"
            "ConfusionMatrixDisplay(matrix).plot(cmap='Blues'); "
            "plt.title(f'Threshold={threshold:.3f}')"
        ),
        markdown("## Capacity curve: the management decision view"),
        code(
            "topk = pd.read_csv(MODEL / 'top_k_business_metrics.csv')\n"
            "topk.style.format({'population_fraction': '{:.0%}', 'precision': '{:.2%}', "
            "'recall': '{:.2%}', 'lift': '{:.3f}'})"
        ),
        code(
            "fig, ax1 = plt.subplots(figsize=(9, 4))\n"
            "ax1.plot(topk['population_fraction'], topk['precision'], marker='o', "
            "label='Precision')\n"
            "ax1.plot(topk['population_fraction'], topk['recall'], marker='o', "
            "label='Recall')\n"
            "ax1.set(xlabel='Population contacted', ylabel='Rate', title='Capacity trade-off')\n"
            "ax1.legend(); ax1.grid(alpha=.3); plt.tight_layout()"
        ),
        markdown("## Stability by test snapshot"),
        code(
            "stability = pd.read_csv(MODEL / 'test_metrics_by_snapshot.csv')\n"
            "ax = stability.plot(x='snapshot_day', y=['pr_auc','roc_auc','brier_score'], "
            "marker='o', figsize=(10,4), title='Holdout stability across time')\n"
            "ax.grid(alpha=.3); plt.tight_layout()"
        ),
        markdown("## Segment and error analysis"),
        code(
            "segments = pd.read_csv(MODEL / 'test_metrics_by_segment.csv')\n"
            "segments"
        ),
        markdown(
            "Model yếu hơn ở nhóm activity trung bình và long-recency. Nhóm có demographic "
            "có purchase rate cao hơn nhiều, nhiều khả năng do selection bias; không diễn "
            "giải demographic là nguyên nhân."
        ),
        markdown("## Explainability by feature family"),
        code(
            "grouped = pd.read_csv(MODEL / 'grouped_feature_importance.csv')\n"
            "grouped"
        ),
        code(
            "plot_grouped = grouped.sort_values('importance_pr_auc_drop')\n"
            "ax = plot_grouped.plot.barh(x='feature_family', y='importance_pr_auc_drop', "
            "legend=False, color='#5B9BD5', title='Grouped permutation importance')\n"
            "ax.set_xlabel('PR-AUC drop after joint permutation'); plt.tight_layout()"
        ),
        markdown(
            "Cadence và assortment/store diversity chi phối. Individual features tương "
            "quan mạnh giữa các windows nên grouped importance đáng tin hơn cho narrative."
        ),
        markdown("## Drift evidence"),
        code(
            "drift = pd.read_csv(MODEL / 'feature_drift_psi.csv')\n"
            "drift.head(15)"
        ),
        markdown(
            "Tenure/lifetime drift là structural do snapshot về sau tích lũy nhiều lịch sử hơn; "
            "không dùng chúng làm fixed PSI alert. Rolling features mới là monitoring signal."
        ),
        markdown("# Phase 6 – Deployment"),
        markdown("## Latest batch score"),
        code(
            "scores = pd.read_csv(MODEL / 'latest_scores_v2.csv')\n"
            "scores.head(10)"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11,4))\n"
            "scores['purchase_probability_next_7d'].hist(bins=30, ax=axes[0])\n"
            "axes[0].set_title('Latest score distribution')\n"
            "scores.plot.scatter(x='rank', y='purchase_probability_next_7d', ax=axes[1], s=8)\n"
            "axes[1].set_title('Probability by rank'); plt.tight_layout()"
        ),
        markdown(
            "## Production contract\n\n"
            "Output: household, snapshot, calibrated probability, high-propensity flag, "
            "rank và percentile. Monitor input contract, rolling-feature drift, calibration, "
            "PR-AUC/lift khi label mature, cùng incremental business KPIs."
        ),
        markdown(
            "## Recommendation\n\n"
            "**Go:** controlled weekly prioritization pilot.\n\n"
            "**No-go:** auto-coupon optimization cho đến khi có randomized treatment/control.\n\n"
            "Pilot cần đo incremental purchase, contribution margin, contact cost và "
            "unsubscribe/complaint guardrails qua tối thiểu 6–8 chu kỳ."
        ),
        markdown(
            "# Reproducibility\n\n"
            "```powershell\n"
            "python src/audit_dunnhumby_data.py\n"
            "python src/run_propensity_v2.py train\n"
            "python src/run_propensity_v2.py score\n"
            "python src/validate_propensity_v2.py\n"
            "python -m pytest -q\n"
            "```"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path("notebooks/customer_purchase_propensity_crisp_dm_v2.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {output} with {len(cells)} cells")


if __name__ == "__main__":
    main()
