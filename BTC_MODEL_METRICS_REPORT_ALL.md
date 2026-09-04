# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T19:37:15.900567+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1232 | 944 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1108 | 743 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 783 | 505 | 277 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 784 | 558 | 224 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 505 | 243 | 262 | 48.12% | 44.58% | 48.12% | 1.88 pp | -19 | 48 | -0.40 |
| BTC Market Hours | transformer | Transformer | 505 | 241 | 264 | 47.72% | 46.25% | 48.33% | 2.28 pp | -23 | 48 | -0.48 |
| BTC Daily | mlp_sklearn | MLPClassifier | 733 | 355 | 378 | 48.43% | 47.08% | 48.33% | 1.57 pp | -23 | 43 | -0.53 |
| BTC Market Hours | nn | NN | 505 | 239 | 266 | 47.33% | 50.42% | 48.33% | 2.67 pp | -27 | 48 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 558 | 263 | 295 | 47.13% | 50.00% | 47.92% | 2.87 pp | -32 | 48 | -0.67 |
| BTC Daily | transformer | Transformer | 733 | 350 | 383 | 47.75% | 46.67% | 49.58% | 2.25 pp | -33 | 43 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 910 | 436 | 474 | 47.91% | 50.83% | 48.12% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | nn | NN | 558 | 259 | 299 | 46.42% | 45.83% | 47.71% | 3.58 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 558 | 258 | 300 | 46.24% | 49.58% | 46.67% | 3.76 pp | -42 | 48 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 910 | 430 | 480 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 733 | 339 | 394 | 46.25% | 44.58% | 47.08% | 3.75 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 505 | 218 | 287 | 43.17% | 41.67% | 43.33% | 6.83 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 505 | 216 | 289 | 42.77% | 43.75% | 42.92% | 7.23 pp | -73 | 48 | -1.52 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 505 | 208 | 297 | 41.19% | 42.50% | 41.67% | 8.81 pp | -89 | 48 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 558 | 231 | 327 | 41.40% | 42.08% | 40.42% | 8.60 pp | -96 | 48 | -2.00 |
| BTC Hourly | nn | NN | 910 | 405 | 505 | 44.51% | 44.17% | 42.29% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 910 | 405 | 505 | 44.51% | 44.58% | 44.17% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 558 | 225 | 333 | 40.32% | 38.75% | 40.42% | 9.68 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 558 | 223 | 335 | 39.96% | 40.83% | 38.96% | 10.04 pp | -112 | 48 | -2.33 |
| BTC Daily | lstm | LSTM | 733 | 315 | 418 | 42.97% | 36.67% | 41.25% | 7.03 pp | -103 | 43 | -2.40 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| BTC Daily | rf | RandomForest | 733 | 312 | 421 | 42.56% | 40.42% | 43.33% | 7.44 pp | -109 | 43 | -2.53 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 910 | 389 | 521 | 42.75% | 39.58% | 41.67% | 7.25 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 910 | 382 | 528 | 41.98% | 41.25% | 40.83% | 8.02 pp | -146 | 48 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 743 | 293 | 450 | 39.43% | 35.83% | 38.12% | 10.57 pp | -157 | 43 | -3.65 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 910 | 436 | 474 | 47.91% | 50.83% | 48.12% | 2.09 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 910 | 430 | 480 | 47.25% | 47.92% | 46.67% | 2.75 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 910 | 405 | 505 | 44.51% | 44.17% | 42.29% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 910 | 405 | 505 | 44.51% | 44.58% | 44.17% | 5.49 pp | -100 | 48 | -2.08 |
| BTC Hourly | lstm | LSTM | 910 | 389 | 521 | 42.75% | 39.58% | 41.67% | 7.25 pp | -132 | 48 | -2.75 |
| BTC Hourly | xgb | XGBoost | 910 | 382 | 528 | 41.98% | 41.25% | 40.83% | 8.02 pp | -146 | 48 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 733 | 355 | 378 | 48.43% | 47.08% | 48.33% | 1.57 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 733 | 350 | 383 | 47.75% | 46.67% | 49.58% | 2.25 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 733 | 339 | 394 | 46.25% | 44.58% | 47.08% | 3.75 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 733 | 315 | 418 | 42.97% | 36.67% | 41.25% | 7.03 pp | -103 | 43 | -2.40 |
| BTC Daily | rf | RandomForest | 733 | 312 | 421 | 42.56% | 40.42% | 43.33% | 7.44 pp | -109 | 43 | -2.53 |
| BTC Daily | xgb | XGBoost | 743 | 293 | 450 | 39.43% | 35.83% | 38.12% | 10.57 pp | -157 | 43 | -3.65 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 505 | 243 | 262 | 48.12% | 44.58% | 48.12% | 1.88 pp | -19 | 48 | -0.40 |
| BTC Market Hours | transformer | Transformer | 505 | 241 | 264 | 47.72% | 46.25% | 48.33% | 2.28 pp | -23 | 48 | -0.48 |
| BTC Market Hours | nn | NN | 505 | 239 | 266 | 47.33% | 50.42% | 48.33% | 2.67 pp | -27 | 48 | -0.56 |
| BTC Market Hours | lstm | LSTM | 505 | 218 | 287 | 43.17% | 41.67% | 43.33% | 6.83 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 505 | 216 | 289 | 42.77% | 43.75% | 42.92% | 7.23 pp | -73 | 48 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 505 | 208 | 297 | 41.19% | 42.50% | 41.67% | 8.81 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 558 | 263 | 295 | 47.13% | 50.00% | 47.92% | 2.87 pp | -32 | 48 | -0.67 |
| BTC Market Hours Daily | nn | NN | 558 | 259 | 299 | 46.42% | 45.83% | 47.71% | 3.58 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 558 | 258 | 300 | 46.24% | 49.58% | 46.67% | 3.76 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 558 | 231 | 327 | 41.40% | 42.08% | 40.42% | 8.60 pp | -96 | 48 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 558 | 225 | 333 | 40.32% | 38.75% | 40.42% | 9.68 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 558 | 223 | 335 | 39.96% | 40.83% | 38.96% | 10.04 pp | -112 | 48 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
