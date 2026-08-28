# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T21:36:52.532269+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 990 | 625 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 576 | 387 | 188 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 20:00:00+00:00 | 578 | 441 | 135 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 387 | 191 | 196 | 49.35% | 47.50% | 49.35% | 0.65 pp | -5 | 39 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 615 | 302 | 313 | 49.11% | 49.17% | 50.00% | 0.89 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 615 | 301 | 314 | 48.94% | 47.50% | 50.42% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Market Hours | nn | NN | 387 | 180 | 207 | 46.51% | 48.75% | 46.51% | 3.49 pp | -27 | 39 | -0.69 |
| BTC Market Hours | transformer | Transformer | 387 | 180 | 207 | 46.51% | 44.17% | 46.51% | 3.49 pp | -27 | 39 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 441 | 203 | 238 | 46.03% | 47.92% | 46.03% | 3.97 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 441 | 202 | 239 | 45.80% | 46.25% | 45.80% | 4.20 pp | -37 | 39 | -0.95 |
| Consolidated Hourly | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 615 | 288 | 327 | 46.83% | 42.92% | 48.75% | 3.17 pp | -39 | 38 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 793 | 374 | 419 | 47.16% | 45.00% | 46.88% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 793 | 374 | 419 | 47.16% | 44.17% | 46.46% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | nn | NN | 441 | 200 | 241 | 45.35% | 46.25% | 45.35% | 4.65 pp | -41 | 39 | -1.05 |
| BTC Market Hours | lstm | LSTM | 387 | 167 | 220 | 43.15% | 43.75% | 43.15% | 6.85 pp | -53 | 39 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 387 | 166 | 221 | 42.89% | 40.83% | 42.89% | 7.11 pp | -55 | 39 | -1.41 |
| BTC Daily | lstm | LSTM | 615 | 275 | 340 | 44.72% | 43.75% | 44.58% | 5.28 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 793 | 357 | 436 | 45.02% | 40.42% | 45.21% | 4.98 pp | -79 | 43 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 387 | 157 | 230 | 40.57% | 38.75% | 40.57% | 9.43 pp | -73 | 39 | -1.87 |
| BTC Hourly | rf | RandomForest | 793 | 352 | 441 | 44.39% | 42.50% | 43.75% | 5.61 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 441 | 180 | 261 | 40.82% | 40.00% | 40.82% | 9.18 pp | -81 | 39 | -2.08 |
| BTC Hourly | lstm | LSTM | 793 | 350 | 443 | 44.14% | 44.17% | 45.62% | 5.86 pp | -93 | 43 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 441 | 175 | 266 | 39.68% | 37.50% | 39.68% | 10.32 pp | -91 | 39 | -2.33 |
| BTC Daily | rf | RandomForest | 615 | 262 | 353 | 42.60% | 42.08% | 43.33% | 7.40 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 441 | 173 | 268 | 39.23% | 37.50% | 39.23% | 10.77 pp | -95 | 39 | -2.44 |
| BTC Hourly | xgb | XGBoost | 793 | 337 | 456 | 42.50% | 39.17% | 43.96% | 7.50 pp | -119 | 43 | -2.77 |
| Consolidated Hourly | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 5 | -3.00 |
| BTC Daily | xgb | XGBoost | 625 | 248 | 377 | 39.68% | 32.92% | 40.00% | 10.32 pp | -129 | 38 | -3.39 |

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
| BTC Daily | transformer | Transformer | 615 | 302 | 313 | 49.11% | 49.17% | 50.00% | 0.89 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 615 | 301 | 314 | 48.94% | 47.50% | 50.42% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 615 | 288 | 327 | 46.83% | 42.92% | 48.75% | 3.17 pp | -39 | 38 | -1.03 |
| BTC Daily | lstm | LSTM | 615 | 275 | 340 | 44.72% | 43.75% | 44.58% | 5.28 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 615 | 262 | 353 | 42.60% | 42.08% | 43.33% | 7.40 pp | -91 | 38 | -2.39 |
| BTC Daily | xgb | XGBoost | 625 | 248 | 377 | 39.68% | 32.92% | 40.00% | 10.32 pp | -129 | 38 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 387 | 191 | 196 | 49.35% | 47.50% | 49.35% | 0.65 pp | -5 | 39 | -0.13 |
| BTC Market Hours | nn | NN | 387 | 180 | 207 | 46.51% | 48.75% | 46.51% | 3.49 pp | -27 | 39 | -0.69 |
| BTC Market Hours | transformer | Transformer | 387 | 180 | 207 | 46.51% | 44.17% | 46.51% | 3.49 pp | -27 | 39 | -0.69 |
| BTC Market Hours | lstm | LSTM | 387 | 167 | 220 | 43.15% | 43.75% | 43.15% | 6.85 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 387 | 166 | 221 | 42.89% | 40.83% | 42.89% | 7.11 pp | -55 | 39 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 387 | 157 | 230 | 40.57% | 38.75% | 40.57% | 9.43 pp | -73 | 39 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 441 | 203 | 238 | 46.03% | 47.92% | 46.03% | 3.97 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 441 | 202 | 239 | 45.80% | 46.25% | 45.80% | 4.20 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | nn | NN | 441 | 200 | 241 | 45.35% | 46.25% | 45.35% | 4.65 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 441 | 180 | 261 | 40.82% | 40.00% | 40.82% | 9.18 pp | -81 | 39 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 441 | 175 | 266 | 39.68% | 37.50% | 39.68% | 10.32 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 441 | 173 | 268 | 39.23% | 37.50% | 39.23% | 10.77 pp | -95 | 39 | -2.44 |

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
