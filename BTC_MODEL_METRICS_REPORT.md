# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T04:49:38.482795+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1238 | 950 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1114 | 749 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 795 | 511 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 797 | 565 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 158 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 511 | 248 | 263 | 48.53% | 45.83% | 48.54% | 1.47 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 511 | 246 | 265 | 48.14% | 47.08% | 48.54% | 1.86 pp | -19 | 49 | -0.39 |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 739 | 358 | 381 | 48.44% | 47.08% | 48.33% | 1.56 pp | -23 | 44 | -0.52 |
| BTC Market Hours Daily | transformer | Transformer | 565 | 269 | 296 | 47.61% | 51.25% | 48.96% | 2.39 pp | -27 | 49 | -0.55 |
| BTC Market Hours | nn | NN | 511 | 241 | 270 | 47.16% | 50.00% | 48.12% | 2.84 pp | -29 | 49 | -0.59 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 739 | 352 | 387 | 47.63% | 45.83% | 49.38% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 916 | 438 | 478 | 47.82% | 49.58% | 47.71% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | nn | NN | 565 | 262 | 303 | 46.37% | 45.83% | 47.92% | 3.63 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 565 | 261 | 304 | 46.19% | 48.75% | 46.46% | 3.81 pp | -43 | 49 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 916 | 432 | 484 | 47.16% | 47.50% | 46.25% | 2.84 pp | -52 | 48 | -1.08 |
| BTC Daily | nn | NN | 739 | 342 | 397 | 46.28% | 43.75% | 46.67% | 3.72 pp | -55 | 44 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 511 | 222 | 289 | 43.44% | 42.92% | 43.54% | 6.56 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 511 | 221 | 290 | 43.25% | 44.58% | 43.54% | 6.75 pp | -69 | 49 | -1.41 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 511 | 210 | 301 | 41.10% | 42.50% | 41.67% | 8.90 pp | -91 | 49 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 565 | 235 | 330 | 41.59% | 42.08% | 40.62% | 8.41 pp | -95 | 49 | -1.94 |
| Consolidated Hourly | transformer | Transformer | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 12 | -2.08 |
| BTC Hourly | rf | RandomForest | 916 | 407 | 509 | 44.43% | 43.75% | 44.17% | 5.57 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 565 | 230 | 335 | 40.71% | 39.58% | 40.83% | 9.29 pp | -105 | 49 | -2.14 |
| BTC Hourly | nn | NN | 916 | 406 | 510 | 44.32% | 42.50% | 42.08% | 5.68 pp | -104 | 48 | -2.17 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 565 | 227 | 338 | 40.18% | 40.83% | 39.38% | 9.82 pp | -111 | 49 | -2.27 |
| BTC Daily | lstm | LSTM | 739 | 318 | 421 | 43.03% | 37.08% | 41.04% | 6.97 pp | -103 | 44 | -2.34 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 739 | 313 | 426 | 42.35% | 39.58% | 42.71% | 7.65 pp | -113 | 44 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 916 | 391 | 525 | 42.69% | 38.75% | 41.46% | 7.31 pp | -134 | 48 | -2.79 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 916 | 382 | 534 | 41.70% | 39.58% | 40.00% | 8.30 pp | -152 | 48 | -3.17 |
| BTC Daily | xgb | XGBoost | 749 | 296 | 453 | 39.52% | 36.25% | 37.71% | 10.48 pp | -157 | 44 | -3.57 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 916 | 438 | 478 | 47.82% | 49.58% | 47.71% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Hourly | transformer | Transformer | 916 | 432 | 484 | 47.16% | 47.50% | 46.25% | 2.84 pp | -52 | 48 | -1.08 |
| BTC Hourly | rf | RandomForest | 916 | 407 | 509 | 44.43% | 43.75% | 44.17% | 5.57 pp | -102 | 48 | -2.12 |
| BTC Hourly | nn | NN | 916 | 406 | 510 | 44.32% | 42.50% | 42.08% | 5.68 pp | -104 | 48 | -2.17 |
| BTC Hourly | lstm | LSTM | 916 | 391 | 525 | 42.69% | 38.75% | 41.46% | 7.31 pp | -134 | 48 | -2.79 |
| BTC Hourly | xgb | XGBoost | 916 | 382 | 534 | 41.70% | 39.58% | 40.00% | 8.30 pp | -152 | 48 | -3.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 739 | 358 | 381 | 48.44% | 47.08% | 48.33% | 1.56 pp | -23 | 44 | -0.52 |
| BTC Daily | transformer | Transformer | 739 | 352 | 387 | 47.63% | 45.83% | 49.38% | 2.37 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 739 | 342 | 397 | 46.28% | 43.75% | 46.67% | 3.72 pp | -55 | 44 | -1.25 |
| BTC Daily | lstm | LSTM | 739 | 318 | 421 | 43.03% | 37.08% | 41.04% | 6.97 pp | -103 | 44 | -2.34 |
| BTC Daily | rf | RandomForest | 739 | 313 | 426 | 42.35% | 39.58% | 42.71% | 7.65 pp | -113 | 44 | -2.57 |
| BTC Daily | xgb | XGBoost | 749 | 296 | 453 | 39.52% | 36.25% | 37.71% | 10.48 pp | -157 | 44 | -3.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 511 | 248 | 263 | 48.53% | 45.83% | 48.54% | 1.47 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 511 | 246 | 265 | 48.14% | 47.08% | 48.54% | 1.86 pp | -19 | 49 | -0.39 |
| BTC Market Hours | nn | NN | 511 | 241 | 270 | 47.16% | 50.00% | 48.12% | 2.84 pp | -29 | 49 | -0.59 |
| BTC Market Hours | lstm | LSTM | 511 | 222 | 289 | 43.44% | 42.92% | 43.54% | 6.56 pp | -67 | 49 | -1.37 |
| BTC Market Hours | rf | RandomForest | 511 | 221 | 290 | 43.25% | 44.58% | 43.54% | 6.75 pp | -69 | 49 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 511 | 210 | 301 | 41.10% | 42.50% | 41.67% | 8.90 pp | -91 | 49 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 565 | 269 | 296 | 47.61% | 51.25% | 48.96% | 2.39 pp | -27 | 49 | -0.55 |
| BTC Market Hours Daily | nn | NN | 565 | 262 | 303 | 46.37% | 45.83% | 47.92% | 3.63 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 565 | 261 | 304 | 46.19% | 48.75% | 46.46% | 3.81 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 565 | 235 | 330 | 41.59% | 42.08% | 40.62% | 8.41 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 565 | 230 | 335 | 40.71% | 39.58% | 40.83% | 9.29 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 565 | 227 | 338 | 40.18% | 40.83% | 39.38% | 9.82 pp | -111 | 49 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
