# Risk register ? DRAFT, pending review

| Risk | Mitigation / open decision |
| --- | --- |
| Purchase completion cannot be verified | Audit semantics; use explicitly named recorded-order proxy if justified. |
| Missing label partitions | Require full coverage before assigning zero. |
| Timestamp perturbation or leakage | Verify timestamps; enforce 24-hour gap and boundary tests. |
| Biased sample or scarce positives | Sample users before outcomes; audit distributions and uncertainty. |
| Resource and network constraints | Inventory first; stream partitions, cap memory and record costs. |
| License and non-commercial restriction | Attribute source and review CC BY-NC-SA 4.0 terms. |
| Propensity mistaken for uplift | State causal limitation and validate any intervention separately. |
