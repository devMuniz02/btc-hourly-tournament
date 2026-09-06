# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T06:12:36.187934+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1256 | 968 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1131 | 766 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 825 | 528 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 827 | 582 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 528 | 257 | 271 | 48.67% | 45.83% | 48.54% | 1.33 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 528 | 253 | 275 | 47.92% | 47.92% | 48.33% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 756 | 367 | 389 | 48.54% | 47.92% | 48.75% | 1.46 pp | -22 | 44 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 582 | 276 | 306 | 47.42% | 50.83% | 48.54% | 2.58 pp | -30 | 50 | -0.60 |
| BTC Market Hours | nn | NN | 528 | 249 | 279 | 47.16% | 50.83% | 48.54% | 2.84 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 582 | 271 | 311 | 46.56% | 46.25% | 47.71% | 3.44 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 934 | 446 | 488 | 47.75% | 50.00% | 47.29% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 582 | 269 | 313 | 46.22% | 51.25% | 46.88% | 3.78 pp | -44 | 50 | -0.88 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| BTC Daily | transformer | Transformer | 756 | 356 | 400 | 47.09% | 43.33% | 48.33% | 2.91 pp | -44 | 44 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 934 | 438 | 496 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| BTC Daily | nn | NN | 756 | 350 | 406 | 46.30% | 44.58% | 46.25% | 3.70 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 528 | 228 | 300 | 43.18% | 42.08% | 43.96% | 6.82 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 528 | 227 | 301 | 42.99% | 44.58% | 43.54% | 7.01 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 528 | 218 | 310 | 41.29% | 42.92% | 41.67% | 8.71 pp | -92 | 50 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 582 | 241 | 341 | 41.41% | 43.33% | 41.04% | 8.59 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 934 | 416 | 518 | 44.54% | 44.58% | 44.17% | 5.46 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 934 | 414 | 520 | 44.33% | 42.50% | 42.29% | 5.67 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 582 | 236 | 346 | 40.55% | 40.00% | 40.42% | 9.45 pp | -110 | 50 | -2.20 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 582 | 231 | 351 | 39.69% | 40.83% | 38.75% | 10.31 pp | -120 | 50 | -2.40 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 756 | 320 | 436 | 42.33% | 35.42% | 40.42% | 7.67 pp | -116 | 44 | -2.64 |
| BTC Hourly | lstm | LSTM | 934 | 399 | 535 | 42.72% | 37.50% | 41.88% | 7.28 pp | -136 | 49 | -2.78 |
| BTC Daily | rf | RandomForest | 756 | 315 | 441 | 41.67% | 37.92% | 41.46% | 8.33 pp | -126 | 44 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 934 | 393 | 541 | 42.08% | 41.25% | 40.83% | 7.92 pp | -148 | 49 | -3.02 |
| BTC Daily | xgb | XGBoost | 766 | 299 | 467 | 39.03% | 35.00% | 36.67% | 10.97 pp | -168 | 44 | -3.82 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 934 | 446 | 488 | 47.75% | 50.00% | 47.29% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 934 | 438 | 496 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 934 | 416 | 518 | 44.54% | 44.58% | 44.17% | 5.46 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 934 | 414 | 520 | 44.33% | 42.50% | 42.29% | 5.67 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 934 | 399 | 535 | 42.72% | 37.50% | 41.88% | 7.28 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 934 | 393 | 541 | 42.08% | 41.25% | 40.83% | 7.92 pp | -148 | 49 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 756 | 367 | 389 | 48.54% | 47.92% | 48.75% | 1.46 pp | -22 | 44 | -0.50 |
| BTC Daily | transformer | Transformer | 756 | 356 | 400 | 47.09% | 43.33% | 48.33% | 2.91 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 756 | 350 | 406 | 46.30% | 44.58% | 46.25% | 3.70 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 756 | 320 | 436 | 42.33% | 35.42% | 40.42% | 7.67 pp | -116 | 44 | -2.64 |
| BTC Daily | rf | RandomForest | 756 | 315 | 441 | 41.67% | 37.92% | 41.46% | 8.33 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 766 | 299 | 467 | 39.03% | 35.00% | 36.67% | 10.97 pp | -168 | 44 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 528 | 257 | 271 | 48.67% | 45.83% | 48.54% | 1.33 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 528 | 253 | 275 | 47.92% | 47.92% | 48.33% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 528 | 249 | 279 | 47.16% | 50.83% | 48.54% | 2.84 pp | -30 | 50 | -0.60 |
| BTC Market Hours | lstm | LSTM | 528 | 228 | 300 | 43.18% | 42.08% | 43.96% | 6.82 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 528 | 227 | 301 | 42.99% | 44.58% | 43.54% | 7.01 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 528 | 218 | 310 | 41.29% | 42.92% | 41.67% | 8.71 pp | -92 | 50 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 582 | 276 | 306 | 47.42% | 50.83% | 48.54% | 2.58 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 582 | 271 | 311 | 46.56% | 46.25% | 47.71% | 3.44 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 582 | 269 | 313 | 46.22% | 51.25% | 46.88% | 3.78 pp | -44 | 50 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 582 | 241 | 341 | 41.41% | 43.33% | 41.04% | 8.59 pp | -100 | 50 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 582 | 236 | 346 | 40.55% | 40.00% | 40.42% | 9.45 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 582 | 231 | 351 | 39.69% | 40.83% | 38.75% | 10.31 pp | -120 | 50 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
