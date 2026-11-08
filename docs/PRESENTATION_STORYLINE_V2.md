# Presentation Storyline – 12 slides

## Slide 1 – Executive question

**Which households are most likely to purchase in the next 7 days, when weekly
capacity is limited?**

Speaker note: mở bằng decision, không mở bằng thuật toán.

## Slide 2 – Business value hypothesis

- Prioritize scarce CRM/operational capacity.
- Improve buyer concentration in selected audience.
- Reduce unnecessary contacts.
- Foundation for a later incremental-uplift pilot.

Speaker note: nói rõ propensity ≠ uplift ngay từ đầu.

## Slide 3 – CRISP-DM and governance

Hiển thị sáu phase, decision owner, weekly cadence và model/business gates.

## Slide 4 – Data landscape

- 8 tables; 39,5M+ source rows including causal media.
- 2.500 households, 276.484 baskets, 711 days.
- Product join 100%; demographic coverage 32,04%.

## Slide 5 – Data finding that changed the solution

`QUANTITY` up to 89.638, concentrated in fuel/misc sales. V2 removed raw quantity
and used basket, line and diversity features instead.

Speaker note: đây là bằng chứng Data Understanding thực sự ảnh hưởng Modeling.

## Slide 6 – Target and leakage-safe design

- Target: any basket in next 7 days.
- 7-day target 53,05%; 14-day 68,26%; 30-day 80,96%.
- Feature cutoff, label maturity, cohort eligibility and two embargoes.

## Slide 7 – Feature architecture

170 point-in-time features grouped into cadence, basket, monetary, promotion,
category, brand, daypart, diversity, lifecycle and demographics.

## Slide 8 – Model selection, not model shopping

Show rolling backtest table. Explain HGB wins narrowly; logistic remains challenger.

## Slide 9 – Holdout results with uncertainty

- PR-AUC 0,8506 [0,8409, 0,8612].
- ROC-AUC 0,8303 [0,8226, 0,8384].
- Lift@10% 1,819x [1,779, 1,863].
- Calibration endpoints.

## Slide 10 – Operational capacity curve

Show top-5/10/20/30/50% table. Ask management to choose capacity using unit economics.

## Slide 11 – What the model learned

Cadence and store/assortment diversity dominate. Demographic/category/promotion
add limited incremental signal after recent behavior is known.

## Slide 12 – Recommendation and ask

- Approve controlled weekly pilot.
- Confirm action, capacity and unit economics.
- Use randomized holdout for incremental value.
- No auto-couponing until uplift evidence exists.

Closing line: “The model reliably identifies likely buyers; the pilot must now
prove whether the business action changes their behavior profitably.”
