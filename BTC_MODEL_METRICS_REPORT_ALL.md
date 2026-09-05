# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T04:10:44.923159+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1113 | 748 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 794 | 510 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 796 | 564 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 157 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 12:00:00+00:00 | 157 | 41 | 116 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 510 | 247 | 263 | 48.43% | 45.42% | 48.54% | 1.57 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 510 | 245 | 265 | 48.04% | 46.67% | 48.33% | 1.96 pp | -20 | 49 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 738 | 357 | 381 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 44 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 564 | 268 | 296 | 47.52% | 51.25% | 48.75% | 2.48 pp | -28 | 49 | -0.57 |
| BTC Market Hours | nn | NN | 510 | 240 | 270 | 47.06% | 49.58% | 47.92% | 2.94 pp | -30 | 49 | -0.61 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 738 | 351 | 387 | 47.56% | 45.83% | 49.38% | 2.44 pp | -36 | 44 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 916 | 438 | 478 | 47.82% | 49.58% | 47.71% | 2.18 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | nn | NN | 564 | 261 | 303 | 46.28% | 45.83% | 47.71% | 3.72 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 564 | 260 | 304 | 46.10% | 48.75% | 46.46% | 3.90 pp | -44 | 49 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| BTC Hourly | transformer | Transformer | 916 | 432 | 484 | 47.16% | 47.50% | 46.25% | 2.84 pp | -52 | 48 | -1.08 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 738 | 341 | 397 | 46.21% | 43.75% | 46.67% | 3.79 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 510 | 222 | 288 | 43.53% | 43.33% | 43.54% | 6.47 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 510 | 220 | 290 | 43.14% | 44.17% | 43.33% | 6.86 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 510 | 210 | 300 | 41.18% | 42.92% | 41.67% | 8.82 pp | -90 | 49 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 564 | 234 | 330 | 41.49% | 42.08% | 40.42% | 8.51 pp | -96 | 49 | -1.96 |
| BTC Hourly | rf | RandomForest | 916 | 407 | 509 | 44.43% | 43.75% | 44.17% | 5.57 pp | -102 | 48 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 564 | 229 | 335 | 40.60% | 39.58% | 40.83% | 9.40 pp | -106 | 49 | -2.16 |
| BTC Hourly | nn | NN | 916 | 406 | 510 | 44.32% | 42.50% | 42.08% | 5.68 pp | -104 | 48 | -2.17 |
| Consolidated Hourly | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 564 | 226 | 338 | 40.07% | 40.83% | 39.17% | 9.93 pp | -112 | 49 | -2.29 |
| BTC Daily | lstm | LSTM | 738 | 318 | 420 | 43.09% | 37.08% | 41.25% | 6.91 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 738 | 312 | 426 | 42.28% | 39.58% | 42.71% | 7.72 pp | -114 | 44 | -2.59 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 916 | 391 | 525 | 42.69% | 38.75% | 41.46% | 7.31 pp | -134 | 48 | -2.79 |
| BTC Hourly | xgb | XGBoost | 916 | 382 | 534 | 41.70% | 39.58% | 40.00% | 8.30 pp | -152 | 48 | -3.17 |
| BTC Daily | xgb | XGBoost | 748 | 295 | 453 | 39.44% | 35.83% | 37.50% | 10.56 pp | -158 | 44 | -3.59 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 738 | 357 | 381 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 44 | -0.55 |
| BTC Daily | transformer | Transformer | 738 | 351 | 387 | 47.56% | 45.83% | 49.38% | 2.44 pp | -36 | 44 | -0.82 |
| BTC Daily | nn | NN | 738 | 341 | 397 | 46.21% | 43.75% | 46.67% | 3.79 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 738 | 318 | 420 | 43.09% | 37.08% | 41.25% | 6.91 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 738 | 312 | 426 | 42.28% | 39.58% | 42.71% | 7.72 pp | -114 | 44 | -2.59 |
| BTC Daily | xgb | XGBoost | 748 | 295 | 453 | 39.44% | 35.83% | 37.50% | 10.56 pp | -158 | 44 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 510 | 247 | 263 | 48.43% | 45.42% | 48.54% | 1.57 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 510 | 245 | 265 | 48.04% | 46.67% | 48.33% | 1.96 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 510 | 240 | 270 | 47.06% | 49.58% | 47.92% | 2.94 pp | -30 | 49 | -0.61 |
| BTC Market Hours | lstm | LSTM | 510 | 222 | 288 | 43.53% | 43.33% | 43.54% | 6.47 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 510 | 220 | 290 | 43.14% | 44.17% | 43.33% | 6.86 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 510 | 210 | 300 | 41.18% | 42.92% | 41.67% | 8.82 pp | -90 | 49 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 564 | 268 | 296 | 47.52% | 51.25% | 48.75% | 2.48 pp | -28 | 49 | -0.57 |
| BTC Market Hours Daily | nn | NN | 564 | 261 | 303 | 46.28% | 45.83% | 47.71% | 3.72 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 564 | 260 | 304 | 46.10% | 48.75% | 46.46% | 3.90 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 564 | 234 | 330 | 41.49% | 42.08% | 40.42% | 8.51 pp | -96 | 49 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 564 | 229 | 335 | 40.60% | 39.58% | 40.83% | 9.40 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 564 | 226 | 338 | 40.07% | 40.83% | 39.17% | 9.93 pp | -112 | 49 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
