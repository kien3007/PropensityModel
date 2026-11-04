# Business understanding ? DRAFT, pending review (M01)

Rank previously active eligible users by probability of at least one verified Retail purchase/order event in the next seven days, for prioritization. This predicts observed natural behavior, not the causal effect or ROI of an intervention. The initial proposed population is catalog users with Retail or Marketplace activity in [T-31 days, T-1 day); feasibility requires source audit. This is not a claim about all users. Unit: (user, snapshot). Deliver a probability and rank with snapshot and model version.

Proposed evaluation: PR-AUC against prevalence, ROC-AUC, precision and lift at a specified K, Brier score, log loss, calibration, temporal backtests and uncertainty. Proposed success is improvement over prevalence and constant probability baselines and lift above one; thresholds are hypotheses, not observed results. Review with stakeholders before approving M01. Purchase meaning, timezone and source fields remain unverified.
