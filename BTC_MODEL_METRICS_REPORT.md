# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T21:55:41.013137+00:00
Scope: `all`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 827 | 292 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 991 | 626 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 577 | 388 | 188 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 579 | 442 | 135 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 47 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 0 | 47 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 09:00:00+00:00 | 47 | 0 | 47 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 388 | 192 | 196 | 49.48% | 47.92% | 49.48% | 0.52 pp | -4 | 39 | -0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 616 | 303 | 313 | 49.19% | 49.17% | 50.21% | 0.81 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 616 | 302 | 314 | 49.03% | 47.92% | 50.42% | 0.97 pp | -12 | 38 | -0.32 |
| BTC Market Hours | nn | NN | 388 | 181 | 207 | 46.65% | 49.17% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | transformer | Transformer | 388 | 181 | 207 | 46.65% | 44.58% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Daily | nn | NN | 616 | 289 | 327 | 46.92% | 43.33% | 48.96% | 3.08 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 793 | 374 | 419 | 47.16% | 45.00% | 46.88% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 793 | 374 | 419 | 47.16% | 44.17% | 46.46% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 388 | 168 | 220 | 43.30% | 43.75% | 43.30% | 6.70 pp | -52 | 39 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 388 | 166 | 222 | 42.78% | 40.42% | 42.78% | 7.22 pp | -56 | 39 | -1.44 |
| BTC Daily | lstm | LSTM | 616 | 275 | 341 | 44.64% | 43.75% | 44.58% | 5.36 pp | -66 | 38 | -1.74 |
| BTC Hourly | nn | NN | 793 | 357 | 436 | 45.02% | 40.42% | 45.21% | 4.98 pp | -79 | 43 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 388 | 158 | 230 | 40.72% | 38.75% | 40.72% | 9.28 pp | -72 | 39 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Hourly | rf | RandomForest | 793 | 352 | 441 | 44.39% | 42.50% | 43.75% | 5.61 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 793 | 350 | 443 | 44.14% | 44.17% | 45.62% | 5.86 pp | -93 | 43 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Daily | rf | RandomForest | 616 | 263 | 353 | 42.69% | 42.50% | 43.33% | 7.31 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 793 | 337 | 456 | 42.50% | 39.17% | 43.96% | 7.50 pp | -119 | 43 | -2.77 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| BTC Daily | xgb | XGBoost | 626 | 249 | 377 | 39.78% | 33.33% | 40.21% | 10.22 pp | -128 | 38 | -3.37 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 793 | 374 | 419 | 47.16% | 45.00% | 46.88% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 793 | 374 | 419 | 47.16% | 44.17% | 46.46% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | nn | NN | 793 | 357 | 436 | 45.02% | 40.42% | 45.21% | 4.98 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 793 | 352 | 441 | 44.39% | 42.50% | 43.75% | 5.61 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 793 | 350 | 443 | 44.14% | 44.17% | 45.62% | 5.86 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 793 | 337 | 456 | 42.50% | 39.17% | 43.96% | 7.50 pp | -119 | 43 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 616 | 303 | 313 | 49.19% | 49.17% | 50.21% | 0.81 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 616 | 302 | 314 | 49.03% | 47.92% | 50.42% | 0.97 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 616 | 289 | 327 | 46.92% | 43.33% | 48.96% | 3.08 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 616 | 275 | 341 | 44.64% | 43.75% | 44.58% | 5.36 pp | -66 | 38 | -1.74 |
| BTC Daily | rf | RandomForest | 616 | 263 | 353 | 42.69% | 42.50% | 43.33% | 7.31 pp | -90 | 38 | -2.37 |
| BTC Daily | xgb | XGBoost | 626 | 249 | 377 | 39.78% | 33.33% | 40.21% | 10.22 pp | -128 | 38 | -3.37 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 388 | 192 | 196 | 49.48% | 47.92% | 49.48% | 0.52 pp | -4 | 39 | -0.10 |
| BTC Market Hours | nn | NN | 388 | 181 | 207 | 46.65% | 49.17% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | transformer | Transformer | 388 | 181 | 207 | 46.65% | 44.58% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | lstm | LSTM | 388 | 168 | 220 | 43.30% | 43.75% | 43.30% | 6.70 pp | -52 | 39 | -1.33 |
| BTC Market Hours | rf | RandomForest | 388 | 166 | 222 | 42.78% | 40.42% | 42.78% | 7.22 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 388 | 158 | 230 | 40.72% | 38.75% | 40.72% | 9.28 pp | -72 | 39 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
