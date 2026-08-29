# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T04:37:57.933536+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1120 | 832 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 996 | 631 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 586 | 393 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 588 | 447 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 393 | 193 | 200 | 49.11% | 47.92% | 49.11% | 0.89 pp | -7 | 40 | -0.17 |
| BTC Daily | transformer | Transformer | 621 | 306 | 315 | 49.28% | 48.75% | 50.21% | 0.72 pp | -9 | 39 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 621 | 303 | 318 | 48.79% | 47.08% | 50.21% | 1.21 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 393 | 184 | 209 | 46.82% | 49.17% | 46.82% | 3.18 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 393 | 183 | 210 | 46.56% | 43.75% | 46.56% | 3.44 pp | -27 | 40 | -0.68 |
| BTC Market Hours Daily | transformer | Transformer | 447 | 207 | 240 | 46.31% | 48.33% | 46.31% | 3.69 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 447 | 205 | 242 | 45.86% | 45.42% | 45.86% | 4.14 pp | -37 | 40 | -0.93 |
| BTC Daily | nn | NN | 621 | 292 | 329 | 47.02% | 43.33% | 48.96% | 2.98 pp | -37 | 39 | -0.95 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 798 | 377 | 421 | 47.24% | 45.00% | 47.29% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 798 | 376 | 422 | 47.12% | 45.00% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 447 | 202 | 245 | 45.19% | 45.42% | 45.19% | 4.81 pp | -43 | 40 | -1.07 |
| BTC Market Hours | lstm | LSTM | 393 | 171 | 222 | 43.51% | 43.33% | 43.51% | 6.49 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 393 | 167 | 226 | 42.49% | 40.83% | 42.49% | 7.51 pp | -59 | 40 | -1.48 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 621 | 276 | 345 | 44.44% | 43.33% | 44.17% | 5.56 pp | -69 | 39 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 393 | 160 | 233 | 40.71% | 39.17% | 40.71% | 9.29 pp | -73 | 40 | -1.82 |
| BTC Hourly | nn | NN | 798 | 359 | 439 | 44.99% | 41.25% | 45.42% | 5.01 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 798 | 355 | 443 | 44.49% | 43.33% | 44.17% | 5.51 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 447 | 182 | 265 | 40.72% | 40.00% | 40.72% | 9.28 pp | -83 | 40 | -2.08 |
| BTC Hourly | lstm | LSTM | 798 | 352 | 446 | 44.11% | 44.58% | 45.83% | 5.89 pp | -94 | 43 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 447 | 179 | 268 | 40.04% | 38.33% | 40.04% | 9.96 pp | -89 | 40 | -2.23 |
| BTC Daily | rf | RandomForest | 621 | 266 | 355 | 42.83% | 42.92% | 43.75% | 7.17 pp | -89 | 39 | -2.28 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 447 | 176 | 271 | 39.37% | 37.50% | 39.37% | 10.63 pp | -95 | 40 | -2.38 |
| BTC Hourly | xgb | XGBoost | 798 | 340 | 458 | 42.61% | 40.00% | 44.38% | 7.39 pp | -118 | 43 | -2.74 |
| BTC Daily | xgb | XGBoost | 631 | 250 | 381 | 39.62% | 32.92% | 40.21% | 10.38 pp | -131 | 39 | -3.36 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 798 | 377 | 421 | 47.24% | 45.00% | 47.29% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 798 | 376 | 422 | 47.12% | 45.00% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 798 | 359 | 439 | 44.99% | 41.25% | 45.42% | 5.01 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 798 | 355 | 443 | 44.49% | 43.33% | 44.17% | 5.51 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 798 | 352 | 446 | 44.11% | 44.58% | 45.83% | 5.89 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 798 | 340 | 458 | 42.61% | 40.00% | 44.38% | 7.39 pp | -118 | 43 | -2.74 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 621 | 306 | 315 | 49.28% | 48.75% | 50.21% | 0.72 pp | -9 | 39 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 621 | 303 | 318 | 48.79% | 47.08% | 50.21% | 1.21 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 621 | 292 | 329 | 47.02% | 43.33% | 48.96% | 2.98 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 621 | 276 | 345 | 44.44% | 43.33% | 44.17% | 5.56 pp | -69 | 39 | -1.77 |
| BTC Daily | rf | RandomForest | 621 | 266 | 355 | 42.83% | 42.92% | 43.75% | 7.17 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 631 | 250 | 381 | 39.62% | 32.92% | 40.21% | 10.38 pp | -131 | 39 | -3.36 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 393 | 193 | 200 | 49.11% | 47.92% | 49.11% | 0.89 pp | -7 | 40 | -0.17 |
| BTC Market Hours | nn | NN | 393 | 184 | 209 | 46.82% | 49.17% | 46.82% | 3.18 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 393 | 183 | 210 | 46.56% | 43.75% | 46.56% | 3.44 pp | -27 | 40 | -0.68 |
| BTC Market Hours | lstm | LSTM | 393 | 171 | 222 | 43.51% | 43.33% | 43.51% | 6.49 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 393 | 167 | 226 | 42.49% | 40.83% | 42.49% | 7.51 pp | -59 | 40 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 393 | 160 | 233 | 40.71% | 39.17% | 40.71% | 9.29 pp | -73 | 40 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 447 | 207 | 240 | 46.31% | 48.33% | 46.31% | 3.69 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 447 | 205 | 242 | 45.86% | 45.42% | 45.86% | 4.14 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 447 | 202 | 245 | 45.19% | 45.42% | 45.19% | 4.81 pp | -43 | 40 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 447 | 182 | 265 | 40.72% | 40.00% | 40.72% | 9.28 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 447 | 179 | 268 | 40.04% | 38.33% | 40.04% | 9.96 pp | -89 | 40 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 447 | 176 | 271 | 39.37% | 37.50% | 39.37% | 10.63 pp | -95 | 40 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

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
