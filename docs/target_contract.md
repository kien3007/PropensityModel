# Target contract ? DRAFT, pending review and source audit

For snapshot T, use only events strictly before T-24 hours for features and eligibility. The target is 1 for at least one verified eligible Retail event in [T, T+7 days); 0 requires complete observable label coverage and no such event; otherwise it is unknown. Preserve negatives. Sample users deterministically independent of outcomes, retaining their events. Splits must have non-overlapping label windows; fit preprocessing and calibration on appropriate earlier periods only.

The actual event value, completion status, raw schema, timezone, coverage and timestamp mapping are unresolved until audit. If completion cannot be verified, call the outcome recorded Retail order, not completed purchase. No target or schema mapping is implemented here.
