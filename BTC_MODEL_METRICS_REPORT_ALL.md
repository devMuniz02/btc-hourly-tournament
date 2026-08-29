# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T16:22:50.090171+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1129 | 841 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1005 | 640 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 599 | 402 | 196 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 601 | 456 | 143 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 1 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 22:00:00+00:00 | 61 | 1 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 402 | 200 | 202 | 49.75% | 48.75% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Daily | transformer | Transformer | 630 | 309 | 321 | 49.05% | 47.50% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 630 | 308 | 322 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| BTC Market Hours | nn | NN | 402 | 191 | 211 | 47.51% | 51.25% | 47.51% | 2.49 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 402 | 185 | 217 | 46.02% | 42.08% | 46.02% | 3.98 pp | -32 | 40 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 456 | 211 | 245 | 46.27% | 46.25% | 46.27% | 3.73 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 456 | 209 | 247 | 45.83% | 47.08% | 45.83% | 4.17 pp | -38 | 40 | -0.95 |
| BTC Daily | nn | NN | 630 | 296 | 334 | 46.98% | 42.92% | 48.96% | 3.02 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 456 | 208 | 248 | 45.61% | 45.83% | 45.61% | 4.39 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 402 | 178 | 224 | 44.28% | 45.83% | 44.28% | 5.72 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 402 | 173 | 229 | 43.03% | 42.08% | 43.03% | 6.97 pp | -56 | 40 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 630 | 279 | 351 | 44.29% | 42.50% | 43.75% | 5.71 pp | -72 | 39 | -1.85 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 402 | 162 | 240 | 40.30% | 38.33% | 40.30% | 9.70 pp | -78 | 40 | -1.95 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 456 | 187 | 269 | 41.01% | 40.83% | 41.01% | 8.99 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 456 | 186 | 270 | 40.79% | 39.58% | 40.79% | 9.21 pp | -84 | 40 | -2.10 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 630 | 269 | 361 | 42.70% | 42.08% | 43.54% | 7.30 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 456 | 179 | 277 | 39.25% | 37.08% | 39.25% | 10.75 pp | -98 | 40 | -2.45 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| BTC Daily | xgb | XGBoost | 640 | 251 | 389 | 39.22% | 31.67% | 39.38% | 10.78 pp | -138 | 39 | -3.54 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 630 | 309 | 321 | 49.05% | 47.50% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 630 | 308 | 322 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 630 | 296 | 334 | 46.98% | 42.92% | 48.96% | 3.02 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 630 | 279 | 351 | 44.29% | 42.50% | 43.75% | 5.71 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 630 | 269 | 361 | 42.70% | 42.08% | 43.54% | 7.30 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 640 | 251 | 389 | 39.22% | 31.67% | 39.38% | 10.78 pp | -138 | 39 | -3.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 402 | 200 | 202 | 49.75% | 48.75% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Market Hours | nn | NN | 402 | 191 | 211 | 47.51% | 51.25% | 47.51% | 2.49 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 402 | 185 | 217 | 46.02% | 42.08% | 46.02% | 3.98 pp | -32 | 40 | -0.80 |
| BTC Market Hours | lstm | LSTM | 402 | 178 | 224 | 44.28% | 45.83% | 44.28% | 5.72 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 402 | 173 | 229 | 43.03% | 42.08% | 43.03% | 6.97 pp | -56 | 40 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 402 | 162 | 240 | 40.30% | 38.33% | 40.30% | 9.70 pp | -78 | 40 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 456 | 211 | 245 | 46.27% | 46.25% | 46.27% | 3.73 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 456 | 209 | 247 | 45.83% | 47.08% | 45.83% | 4.17 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 456 | 208 | 248 | 45.61% | 45.83% | 45.61% | 4.39 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 456 | 187 | 269 | 41.01% | 40.83% | 41.01% | 8.99 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 456 | 186 | 270 | 40.79% | 39.58% | 40.79% | 9.21 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 456 | 179 | 277 | 39.25% | 37.08% | 39.25% | 10.75 pp | -98 | 40 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
