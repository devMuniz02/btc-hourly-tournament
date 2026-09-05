# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T09:04:01.468217+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1241 | 953 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1117 | 752 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 798 | 514 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 800 | 568 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 160 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 160 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 160 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 161 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 514 | 249 | 265 | 48.44% | 45.83% | 48.33% | 1.56 pp | -16 | 49 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 514 | 247 | 267 | 48.05% | 47.50% | 48.54% | 1.95 pp | -20 | 49 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 742 | 360 | 382 | 48.52% | 47.08% | 48.54% | 1.48 pp | -22 | 44 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 568 | 271 | 297 | 47.71% | 51.67% | 48.96% | 2.29 pp | -26 | 49 | -0.53 |
| BTC Market Hours | nn | NN | 514 | 242 | 272 | 47.08% | 49.58% | 48.12% | 2.92 pp | -30 | 49 | -0.61 |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 742 | 354 | 388 | 47.71% | 46.25% | 49.58% | 2.29 pp | -34 | 44 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 919 | 439 | 480 | 47.77% | 49.17% | 47.29% | 2.23 pp | -41 | 48 | -0.85 |
| BTC Market Hours Daily | nn | NN | 568 | 263 | 305 | 46.30% | 45.42% | 47.50% | 3.70 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 568 | 261 | 307 | 45.95% | 48.75% | 46.04% | 4.05 pp | -46 | 49 | -0.94 |
| Consolidated Hourly | xgb | XGBoost | 160 | 74 | 86 | 46.25% | 46.25% | 46.25% | 3.75 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 160 | 74 | 86 | 46.25% | 46.25% | 46.25% | 3.75 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 919 | 435 | 484 | 47.33% | 47.92% | 46.25% | 2.67 pp | -49 | 48 | -1.02 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 742 | 342 | 400 | 46.09% | 42.92% | 46.67% | 3.91 pp | -58 | 44 | -1.32 |
| BTC Market Hours | lstm | LSTM | 514 | 223 | 291 | 43.39% | 42.08% | 43.75% | 6.61 pp | -68 | 49 | -1.39 |
| BTC Market Hours | rf | RandomForest | 514 | 222 | 292 | 43.19% | 45.00% | 43.54% | 6.81 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 160 | 70 | 90 | 43.75% | 43.75% | 43.75% | 6.25 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 160 | 70 | 90 | 43.75% | 43.75% | 43.75% | 6.25 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 514 | 211 | 303 | 41.05% | 42.92% | 41.88% | 8.95 pp | -92 | 49 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 568 | 237 | 331 | 41.73% | 42.50% | 40.83% | 8.27 pp | -94 | 49 | -1.92 |
| BTC Hourly | rf | RandomForest | 919 | 409 | 510 | 44.50% | 43.75% | 44.17% | 5.50 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 919 | 408 | 511 | 44.40% | 42.92% | 42.29% | 5.60 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 568 | 231 | 337 | 40.67% | 40.00% | 40.83% | 9.33 pp | -106 | 49 | -2.16 |
| Consolidated Hourly | transformer | Transformer | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 568 | 227 | 341 | 39.96% | 40.83% | 38.96% | 10.04 pp | -114 | 49 | -2.33 |
| BTC Daily | lstm | LSTM | 742 | 318 | 424 | 42.86% | 36.67% | 40.83% | 7.14 pp | -106 | 44 | -2.41 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 742 | 313 | 429 | 42.18% | 39.17% | 42.50% | 7.82 pp | -116 | 44 | -2.64 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 919 | 393 | 526 | 42.76% | 38.75% | 41.67% | 7.24 pp | -133 | 48 | -2.77 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 919 | 383 | 536 | 41.68% | 38.75% | 39.79% | 8.32 pp | -153 | 48 | -3.19 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 752 | 297 | 455 | 39.49% | 36.25% | 37.71% | 10.51 pp | -158 | 44 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 919 | 439 | 480 | 47.77% | 49.17% | 47.29% | 2.23 pp | -41 | 48 | -0.85 |
| BTC Hourly | transformer | Transformer | 919 | 435 | 484 | 47.33% | 47.92% | 46.25% | 2.67 pp | -49 | 48 | -1.02 |
| BTC Hourly | rf | RandomForest | 919 | 409 | 510 | 44.50% | 43.75% | 44.17% | 5.50 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 919 | 408 | 511 | 44.40% | 42.92% | 42.29% | 5.60 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 919 | 393 | 526 | 42.76% | 38.75% | 41.67% | 7.24 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 919 | 383 | 536 | 41.68% | 38.75% | 39.79% | 8.32 pp | -153 | 48 | -3.19 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 742 | 360 | 382 | 48.52% | 47.08% | 48.54% | 1.48 pp | -22 | 44 | -0.50 |
| BTC Daily | transformer | Transformer | 742 | 354 | 388 | 47.71% | 46.25% | 49.58% | 2.29 pp | -34 | 44 | -0.77 |
| BTC Daily | nn | NN | 742 | 342 | 400 | 46.09% | 42.92% | 46.67% | 3.91 pp | -58 | 44 | -1.32 |
| BTC Daily | lstm | LSTM | 742 | 318 | 424 | 42.86% | 36.67% | 40.83% | 7.14 pp | -106 | 44 | -2.41 |
| BTC Daily | rf | RandomForest | 742 | 313 | 429 | 42.18% | 39.17% | 42.50% | 7.82 pp | -116 | 44 | -2.64 |
| BTC Daily | xgb | XGBoost | 752 | 297 | 455 | 39.49% | 36.25% | 37.71% | 10.51 pp | -158 | 44 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 514 | 249 | 265 | 48.44% | 45.83% | 48.33% | 1.56 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 514 | 247 | 267 | 48.05% | 47.50% | 48.54% | 1.95 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 514 | 242 | 272 | 47.08% | 49.58% | 48.12% | 2.92 pp | -30 | 49 | -0.61 |
| BTC Market Hours | lstm | LSTM | 514 | 223 | 291 | 43.39% | 42.08% | 43.75% | 6.61 pp | -68 | 49 | -1.39 |
| BTC Market Hours | rf | RandomForest | 514 | 222 | 292 | 43.19% | 45.00% | 43.54% | 6.81 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 514 | 211 | 303 | 41.05% | 42.92% | 41.88% | 8.95 pp | -92 | 49 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 568 | 271 | 297 | 47.71% | 51.67% | 48.96% | 2.29 pp | -26 | 49 | -0.53 |
| BTC Market Hours Daily | nn | NN | 568 | 263 | 305 | 46.30% | 45.42% | 47.50% | 3.70 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 568 | 261 | 307 | 45.95% | 48.75% | 46.04% | 4.05 pp | -46 | 49 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 568 | 237 | 331 | 41.73% | 42.50% | 40.83% | 8.27 pp | -94 | 49 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 568 | 231 | 337 | 40.67% | 40.00% | 40.83% | 9.33 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 568 | 227 | 341 | 39.96% | 40.83% | 38.96% | 10.04 pp | -114 | 49 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 160 | 74 | 86 | 46.25% | 46.25% | 46.25% | 3.75 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 160 | 70 | 90 | 43.75% | 43.75% | 43.75% | 6.25 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 160 | 74 | 86 | 46.25% | 46.25% | 46.25% | 3.75 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 160 | 70 | 90 | 43.75% | 43.75% | 43.75% | 6.25 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
