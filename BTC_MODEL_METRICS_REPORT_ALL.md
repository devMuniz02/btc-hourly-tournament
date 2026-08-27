# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T06:10:29.360101+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 797 | 322 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 960 | 595 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 524 | 357 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 526 | 411 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 0 | 21 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 0 | 21 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 357 | 177 | 180 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Daily | transformer | Transformer | 585 | 290 | 295 | 49.57% | 51.67% | 50.00% | 0.43 pp | -5 | 37 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 585 | 286 | 299 | 48.89% | 47.92% | 49.38% | 1.11 pp | -13 | 37 | -0.35 |
| BTC Market Hours | transformer | Transformer | 357 | 169 | 188 | 47.34% | 46.25% | 47.34% | 2.66 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 357 | 163 | 194 | 45.66% | 47.92% | 45.66% | 4.34 pp | -31 | 37 | -0.84 |
| BTC Daily | nn | NN | 585 | 276 | 309 | 47.18% | 45.42% | 48.12% | 2.82 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 411 | 189 | 222 | 45.99% | 47.08% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 411 | 189 | 222 | 45.99% | 48.33% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 411 | 188 | 223 | 45.74% | 46.25% | 45.74% | 4.26 pp | -35 | 37 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 357 | 155 | 202 | 43.42% | 43.33% | 43.42% | 6.58 pp | -47 | 37 | -1.27 |
| BTC Market Hours | rf | RandomForest | 357 | 153 | 204 | 42.86% | 42.08% | 42.86% | 7.14 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 585 | 264 | 321 | 45.13% | 45.83% | 45.21% | 4.87 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 357 | 147 | 210 | 41.18% | 42.08% | 41.18% | 8.82 pp | -63 | 37 | -1.70 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 411 | 169 | 242 | 41.12% | 40.42% | 41.12% | 8.88 pp | -73 | 37 | -1.97 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 585 | 252 | 333 | 43.08% | 43.75% | 43.75% | 6.92 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 411 | 165 | 246 | 40.15% | 38.33% | 40.15% | 9.85 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 37 | -2.24 |
| Consolidated Hourly | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 595 | 240 | 355 | 40.34% | 35.83% | 40.83% | 9.66 pp | -115 | 37 | -3.11 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 585 | 290 | 295 | 49.57% | 51.67% | 50.00% | 0.43 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 585 | 286 | 299 | 48.89% | 47.92% | 49.38% | 1.11 pp | -13 | 37 | -0.35 |
| BTC Daily | nn | NN | 585 | 276 | 309 | 47.18% | 45.42% | 48.12% | 2.82 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 585 | 264 | 321 | 45.13% | 45.83% | 45.21% | 4.87 pp | -57 | 37 | -1.54 |
| BTC Daily | rf | RandomForest | 585 | 252 | 333 | 43.08% | 43.75% | 43.75% | 6.92 pp | -81 | 37 | -2.19 |
| BTC Daily | xgb | XGBoost | 595 | 240 | 355 | 40.34% | 35.83% | 40.83% | 9.66 pp | -115 | 37 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 357 | 177 | 180 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Market Hours | transformer | Transformer | 357 | 169 | 188 | 47.34% | 46.25% | 47.34% | 2.66 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 357 | 163 | 194 | 45.66% | 47.92% | 45.66% | 4.34 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 357 | 155 | 202 | 43.42% | 43.33% | 43.42% | 6.58 pp | -47 | 37 | -1.27 |
| BTC Market Hours | rf | RandomForest | 357 | 153 | 204 | 42.86% | 42.08% | 42.86% | 7.14 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 357 | 147 | 210 | 41.18% | 42.08% | 41.18% | 8.82 pp | -63 | 37 | -1.70 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 411 | 189 | 222 | 45.99% | 47.08% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 411 | 189 | 222 | 45.99% | 48.33% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 411 | 188 | 223 | 45.74% | 46.25% | 45.74% | 4.26 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 411 | 169 | 242 | 41.12% | 40.42% | 41.12% | 8.88 pp | -73 | 37 | -1.97 |
| BTC Market Hours Daily | xgb | XGBoost | 411 | 165 | 246 | 40.15% | 38.33% | 40.15% | 9.85 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 37 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |

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
