# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T01:03:52.019928+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1136 | 848 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1012 | 647 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 615 | 409 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 617 | 463 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 64 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 04:00:00+00:00 | 64 | 0 | 64 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 409 | 203 | 206 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 637 | 312 | 325 | 48.98% | 47.50% | 50.00% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 637 | 309 | 328 | 48.51% | 45.42% | 49.79% | 1.49 pp | -19 | 39 | -0.49 |
| BTC Market Hours | nn | NN | 409 | 193 | 216 | 47.19% | 50.42% | 47.19% | 2.81 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 409 | 189 | 220 | 46.21% | 42.50% | 46.21% | 3.79 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 463 | 214 | 249 | 46.22% | 45.83% | 46.22% | 3.78 pp | -35 | 41 | -0.85 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| BTC Market Hours Daily | transformer | Transformer | 463 | 213 | 250 | 46.00% | 46.67% | 46.00% | 4.00 pp | -37 | 41 | -0.90 |
| BTC Daily | nn | NN | 637 | 300 | 337 | 47.10% | 43.33% | 49.17% | 2.90 pp | -37 | 39 | -0.95 |
| BTC Hourly | transformer | Transformer | 814 | 385 | 429 | 47.30% | 45.83% | 46.25% | 2.70 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | nn | NN | 463 | 211 | 252 | 45.57% | 45.00% | 45.57% | 4.43 pp | -41 | 41 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 814 | 382 | 432 | 46.93% | 43.33% | 46.88% | 3.07 pp | -50 | 44 | -1.14 |
| BTC Market Hours | lstm | LSTM | 409 | 180 | 229 | 44.01% | 45.00% | 44.01% | 5.99 pp | -49 | 41 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 409 | 175 | 234 | 42.79% | 42.08% | 42.79% | 7.21 pp | -59 | 41 | -1.44 |
| BTC Hourly | nn | NN | 814 | 367 | 447 | 45.09% | 41.25% | 45.00% | 4.91 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 637 | 282 | 355 | 44.27% | 42.08% | 43.75% | 5.73 pp | -73 | 39 | -1.87 |
| BTC Hourly | rf | RandomForest | 814 | 364 | 450 | 44.72% | 44.58% | 44.38% | 5.28 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 409 | 164 | 245 | 40.10% | 38.75% | 40.10% | 9.90 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 463 | 190 | 273 | 41.04% | 41.67% | 41.04% | 8.96 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 41 | -2.07 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |
| BTC Hourly | lstm | LSTM | 814 | 355 | 459 | 43.61% | 42.08% | 44.38% | 6.39 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 637 | 272 | 365 | 42.70% | 41.67% | 43.54% | 7.30 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 463 | 181 | 282 | 39.09% | 36.25% | 39.09% | 10.91 pp | -101 | 41 | -2.46 |
| BTC Hourly | xgb | XGBoost | 814 | 345 | 469 | 42.38% | 39.58% | 42.71% | 7.62 pp | -124 | 44 | -2.82 |
| BTC Daily | xgb | XGBoost | 647 | 253 | 394 | 39.10% | 30.83% | 38.96% | 10.90 pp | -141 | 39 | -3.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 814 | 385 | 429 | 47.30% | 45.83% | 46.25% | 2.70 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 814 | 382 | 432 | 46.93% | 43.33% | 46.88% | 3.07 pp | -50 | 44 | -1.14 |
| BTC Hourly | nn | NN | 814 | 367 | 447 | 45.09% | 41.25% | 45.00% | 4.91 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 814 | 364 | 450 | 44.72% | 44.58% | 44.38% | 5.28 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 814 | 355 | 459 | 43.61% | 42.08% | 44.38% | 6.39 pp | -104 | 44 | -2.36 |
| BTC Hourly | xgb | XGBoost | 814 | 345 | 469 | 42.38% | 39.58% | 42.71% | 7.62 pp | -124 | 44 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 637 | 312 | 325 | 48.98% | 47.50% | 50.00% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 637 | 309 | 328 | 48.51% | 45.42% | 49.79% | 1.49 pp | -19 | 39 | -0.49 |
| BTC Daily | nn | NN | 637 | 300 | 337 | 47.10% | 43.33% | 49.17% | 2.90 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 637 | 282 | 355 | 44.27% | 42.08% | 43.75% | 5.73 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 637 | 272 | 365 | 42.70% | 41.67% | 43.54% | 7.30 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 647 | 253 | 394 | 39.10% | 30.83% | 38.96% | 10.90 pp | -141 | 39 | -3.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 409 | 203 | 206 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 409 | 193 | 216 | 47.19% | 50.42% | 47.19% | 2.81 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 409 | 189 | 220 | 46.21% | 42.50% | 46.21% | 3.79 pp | -31 | 41 | -0.76 |
| BTC Market Hours | lstm | LSTM | 409 | 180 | 229 | 44.01% | 45.00% | 44.01% | 5.99 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 409 | 175 | 234 | 42.79% | 42.08% | 42.79% | 7.21 pp | -59 | 41 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 409 | 164 | 245 | 40.10% | 38.75% | 40.10% | 9.90 pp | -81 | 41 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 463 | 214 | 249 | 46.22% | 45.83% | 46.22% | 3.78 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 463 | 213 | 250 | 46.00% | 46.67% | 46.00% | 4.00 pp | -37 | 41 | -0.90 |
| BTC Market Hours Daily | nn | NN | 463 | 211 | 252 | 45.57% | 45.00% | 45.57% | 4.43 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 463 | 190 | 273 | 41.04% | 41.67% | 41.04% | 8.96 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 463 | 181 | 282 | 39.09% | 36.25% | 39.09% | 10.91 pp | -101 | 41 | -2.46 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 64 | 38 | 26 | 59.38% | 59.38% | 59.38% | 9.38 pp | 12 | 7 | 1.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 64 | 34 | 30 | 53.12% | 53.12% | 53.12% | 3.12 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 7 | -2.29 |

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
