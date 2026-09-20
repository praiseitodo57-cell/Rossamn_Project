# Rossmann Store Sales Forecasting

End-to-end machine learning pipeline predicting daily sales for 1,115 Rossmann drug stores, built from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales). Scored using **RMSPE** (Root Mean Squared Percentage Error), the competition's actual evaluation metric.

## Results

| Model | Validation RMSPE |
|---|---|
| Baseline (per-store historical average) | 0.3305 |
| Random Forest (tuned depth) | 0.1722 |
| **XGBoost (final model)** | **0.1319** |

Final model roughly **60% more accurate** than the naive baseline, validated with a strict time-based split to simulate real forecasting conditions.

## Why RMSPE, and why it drove every design decision

RMSPE measures *relative* error, not absolute — critical here because store sizes vary wildly (some stores do 2,000/day, others 30,000/day). A model optimized for RMSE alone would over-focus on large stores. This single choice shaped several downstream decisions in this project:

- The target (`Sales`) is **log-transformed** before training (`log1p`), since minimizing error on a log scale approximates minimizing percentage error — a natural pairing with RMSPE.
- The RMSPE function explicitly **excludes rows where actual Sales = 0** (division-by-zero), matching Kaggle's own scoring rule exactly.

## Pipeline overview

1. **EDA** — examined the Sales distribution (right-skewed), diagnosed *why* values were missing in `store.csv` (some missingness was structural/expected, some was genuinely unknown data), and caught a statistical trap: Sunday appeared to have the "best" average sales, but that was a small-sample artifact from the tiny number of stores open on Sundays.
2. **Merging** — joined `store.csv` into `train`/`test` on `Store`, with explicit checks for duplicate keys (which would silently inflate row counts) and unmatched keys (which show up as missing values after a left join).
3. **Feature engineering** — built date-derived features (Year/Month/Week), a `CompetitionOpen` duration feature (with an explicit "unknown" flag rather than quietly faking a value for missing competitor-open dates), and a `Promo2Active` flag requiring both a month-match against `PromoInterval` *and* a date check against `Promo2Since`. Categorical variables (`StateHoliday`, `StoreType`, `Assortment`) were one-hot encoded using `sklearn.OneHotEncoder` fit only on train, to guarantee train/test column alignment even though `test.csv`'s 6-week window doesn't contain every category `train` does (e.g., no Christmas/Easter holidays fall in the test period).
4. **Validation strategy** — a strictly **time-based** train/validation split (never random shuffling), holding out the most recent 6 weeks of training data to mirror the real test period's length and position in time.
5. **Baseline** — a deliberately naive per-store average, establishing the number every real model had to beat.
6. **Modeling** — Random Forest first, then XGBoost (which won by ~23% relative RMSPE improvement via sequential error-correction rather than independent averaging).
7. **Submission** — predictions inverse-transformed from log scale (`expm1`), with closed-store days (`Open=0`) forced to a predicted 0, matching the competition's rule.

## Debugging notes 

A few real bugs surfaced and were fixed methodically rather than patched blindly:

- **Data leakage check**: `Customers` was excluded from features entirely — it's not available in `test.csv` (you wouldn't know daily footfall before the day happens), so training on it would have produced a model that looked great in validation but couldn't actually run at inference time.
- **Underfitting diagnosis**: an early Random Forest (`max_depth=10`) barely beat the baseline and showed *worse* RMSPE on training data than validation — an unusual pattern. A first hypothesis (holiday days driving the gap) was tested directly against the data and rejected. The real cause was confirmed by increasing tree depth: validation RMSPE dropped from 0.32 to 0.17.
- **Tuning that made things worse**: an attempt to regularize XGBoost (lower learning rate + `min_child_weight` + `gamma`) successfully closed the train/validation gap — but actual validation RMSPE got *worse* (0.1378 vs 0.1319). Lesson: a smaller train/val gap is not itself the goal; the validation metric is.
- **Stale-state submission bug**: a first submission attempt produced 35,104 unexplained `NaN` values. Root cause: assigning predictions and then overriding closed-store rows ran as separate, non-atomic steps across notebook cells, with a stale `Sales` column in between. Fixed by running predict → assign → override → export as a single atomic block.

## Tech stack

`pandas`, `numpy`, `scikit-learn` (Random Forest, OneHotEncoder, SimpleImputer), `xgboost`

## Files

- `rossmann_submission.csv` — final predictions in Kaggle submission format
- Notebook — full pipeline, EDA visualizations, and debugging walkthroughs
