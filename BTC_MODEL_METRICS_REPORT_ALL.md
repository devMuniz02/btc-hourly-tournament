# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T13:42:39.189277+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1146 | 858 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1022 | 657 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 626 | 419 | 206 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 628 | 473 | 153 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 419 | 207 | 212 | 49.40% | 47.50% | 49.40% | 0.60 pp | -5 | 42 | -0.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 647 | 315 | 332 | 48.69% | 46.25% | 50.00% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 647 | 314 | 333 | 48.53% | 45.83% | 49.38% | 1.47 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 419 | 197 | 222 | 47.02% | 50.42% | 47.02% | 2.98 pp | -25 | 42 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 473 | 219 | 254 | 46.30% | 47.08% | 46.30% | 3.70 pp | -35 | 42 | -0.83 |
| BTC Market Hours | transformer | Transformer | 419 | 191 | 228 | 45.58% | 40.83% | 45.58% | 4.42 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 824 | 391 | 433 | 47.45% | 47.08% | 46.67% | 2.55 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 473 | 215 | 258 | 45.45% | 44.58% | 45.45% | 4.55 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 473 | 215 | 258 | 45.45% | 45.00% | 45.45% | 4.55 pp | -43 | 42 | -1.02 |
| BTC Daily | nn | NN | 647 | 303 | 344 | 46.83% | 42.08% | 49.17% | 3.17 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 824 | 386 | 438 | 46.84% | 42.50% | 46.46% | 3.16 pp | -52 | 44 | -1.18 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 419 | 183 | 236 | 43.68% | 43.33% | 43.68% | 6.32 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 419 | 181 | 238 | 43.20% | 42.92% | 43.20% | 6.80 pp | -57 | 42 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 824 | 372 | 452 | 45.15% | 42.50% | 44.79% | 4.85 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 647 | 286 | 361 | 44.20% | 42.08% | 43.75% | 5.80 pp | -75 | 40 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 473 | 196 | 277 | 41.44% | 42.50% | 41.44% | 8.56 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 824 | 368 | 456 | 44.66% | 44.17% | 44.38% | 5.34 pp | -88 | 44 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 37.92% | 39.86% | 10.14 pp | -85 | 42 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 473 | 192 | 281 | 40.59% | 38.75% | 40.59% | 9.41 pp | -89 | 42 | -2.12 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 647 | 276 | 371 | 42.66% | 41.67% | 43.33% | 7.34 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 824 | 359 | 465 | 43.57% | 41.67% | 43.96% | 6.43 pp | -106 | 44 | -2.41 |
| BTC Market Hours Daily | xgb | XGBoost | 473 | 184 | 289 | 38.90% | 35.83% | 38.90% | 11.10 pp | -105 | 42 | -2.50 |
| BTC Hourly | xgb | XGBoost | 824 | 347 | 477 | 42.11% | 39.17% | 42.29% | 7.89 pp | -130 | 44 | -2.95 |
| BTC Daily | xgb | XGBoost | 657 | 259 | 398 | 39.42% | 32.08% | 39.58% | 10.58 pp | -139 | 40 | -3.48 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 824 | 391 | 433 | 47.45% | 47.08% | 46.67% | 2.55 pp | -42 | 44 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 824 | 386 | 438 | 46.84% | 42.50% | 46.46% | 3.16 pp | -52 | 44 | -1.18 |
| BTC Hourly | nn | NN | 824 | 372 | 452 | 45.15% | 42.50% | 44.79% | 4.85 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 824 | 368 | 456 | 44.66% | 44.17% | 44.38% | 5.34 pp | -88 | 44 | -2.00 |
| BTC Hourly | lstm | LSTM | 824 | 359 | 465 | 43.57% | 41.67% | 43.96% | 6.43 pp | -106 | 44 | -2.41 |
| BTC Hourly | xgb | XGBoost | 824 | 347 | 477 | 42.11% | 39.17% | 42.29% | 7.89 pp | -130 | 44 | -2.95 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 647 | 315 | 332 | 48.69% | 46.25% | 50.00% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 647 | 314 | 333 | 48.53% | 45.83% | 49.38% | 1.47 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 647 | 303 | 344 | 46.83% | 42.08% | 49.17% | 3.17 pp | -41 | 40 | -1.02 |
| BTC Daily | lstm | LSTM | 647 | 286 | 361 | 44.20% | 42.08% | 43.75% | 5.80 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 647 | 276 | 371 | 42.66% | 41.67% | 43.33% | 7.34 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 657 | 259 | 398 | 39.42% | 32.08% | 39.58% | 10.58 pp | -139 | 40 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 419 | 207 | 212 | 49.40% | 47.50% | 49.40% | 0.60 pp | -5 | 42 | -0.12 |
| BTC Market Hours | nn | NN | 419 | 197 | 222 | 47.02% | 50.42% | 47.02% | 2.98 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 419 | 191 | 228 | 45.58% | 40.83% | 45.58% | 4.42 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 419 | 183 | 236 | 43.68% | 43.33% | 43.68% | 6.32 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 419 | 181 | 238 | 43.20% | 42.92% | 43.20% | 6.80 pp | -57 | 42 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 37.92% | 39.86% | 10.14 pp | -85 | 42 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 473 | 219 | 254 | 46.30% | 47.08% | 46.30% | 3.70 pp | -35 | 42 | -0.83 |
| BTC Market Hours Daily | nn | NN | 473 | 215 | 258 | 45.45% | 44.58% | 45.45% | 4.55 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 473 | 215 | 258 | 45.45% | 45.00% | 45.45% | 4.55 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 473 | 196 | 277 | 41.44% | 42.50% | 41.44% | 8.56 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 473 | 192 | 281 | 40.59% | 38.75% | 40.59% | 9.41 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 473 | 184 | 289 | 38.90% | 35.83% | 38.90% | 11.10 pp | -105 | 42 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

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
