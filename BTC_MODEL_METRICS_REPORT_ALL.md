# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T10:58:38.848152+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1259 | 971 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1135 | 770 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 829 | 532 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 831 | 586 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 177 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 177 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 51 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 51 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 532 | 259 | 273 | 48.68% | 45.42% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 532 | 255 | 277 | 47.93% | 48.75% | 48.75% | 2.07 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 760 | 370 | 390 | 48.68% | 48.33% | 48.75% | 1.32 pp | -20 | 44 | -0.45 |
| BTC Market Hours | nn | NN | 532 | 253 | 279 | 47.56% | 51.25% | 49.17% | 2.44 pp | -26 | 50 | -0.52 |
| BTC Market Hours Daily | transformer | Transformer | 586 | 278 | 308 | 47.44% | 50.83% | 48.54% | 2.56 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 586 | 273 | 313 | 46.59% | 46.67% | 47.92% | 3.41 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 586 | 272 | 314 | 46.42% | 51.25% | 47.29% | 3.58 pp | -42 | 50 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 937 | 447 | 490 | 47.71% | 50.00% | 47.08% | 2.29 pp | -43 | 49 | -0.88 |
| BTC Daily | transformer | Transformer | 760 | 358 | 402 | 47.11% | 43.33% | 48.12% | 2.89 pp | -44 | 44 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 937 | 440 | 497 | 46.96% | 46.25% | 45.42% | 3.04 pp | -57 | 49 | -1.16 |
| Consolidated Market Hours | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 760 | 352 | 408 | 46.32% | 45.00% | 46.25% | 3.68 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 532 | 230 | 302 | 43.23% | 42.50% | 44.17% | 6.77 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 532 | 229 | 303 | 43.05% | 43.75% | 43.75% | 6.95 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 532 | 221 | 311 | 41.54% | 42.92% | 42.08% | 8.46 pp | -90 | 50 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 586 | 243 | 343 | 41.47% | 44.17% | 41.25% | 8.53 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 937 | 417 | 520 | 44.50% | 44.17% | 43.96% | 5.50 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 937 | 415 | 522 | 44.29% | 42.08% | 42.08% | 5.71 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 586 | 238 | 348 | 40.61% | 40.00% | 40.42% | 9.39 pp | -110 | 50 | -2.20 |
| Consolidated Hourly | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 586 | 233 | 353 | 39.76% | 41.25% | 38.96% | 10.24 pp | -120 | 50 | -2.40 |
| BTC Daily | lstm | LSTM | 760 | 322 | 438 | 42.37% | 35.83% | 40.62% | 7.63 pp | -116 | 44 | -2.64 |
| Consolidated Market Hours | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 937 | 400 | 537 | 42.69% | 37.08% | 41.67% | 7.31 pp | -137 | 49 | -2.80 |
| BTC Daily | rf | RandomForest | 760 | 317 | 443 | 41.71% | 37.92% | 41.88% | 8.29 pp | -126 | 44 | -2.86 |
| BTC Hourly | xgb | XGBoost | 937 | 394 | 543 | 42.05% | 41.25% | 40.62% | 7.95 pp | -149 | 49 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 770 | 303 | 467 | 39.35% | 35.83% | 37.08% | 10.65 pp | -164 | 44 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 937 | 447 | 490 | 47.71% | 50.00% | 47.08% | 2.29 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 937 | 440 | 497 | 46.96% | 46.25% | 45.42% | 3.04 pp | -57 | 49 | -1.16 |
| BTC Hourly | rf | RandomForest | 937 | 417 | 520 | 44.50% | 44.17% | 43.96% | 5.50 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 937 | 415 | 522 | 44.29% | 42.08% | 42.08% | 5.71 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 937 | 400 | 537 | 42.69% | 37.08% | 41.67% | 7.31 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 937 | 394 | 543 | 42.05% | 41.25% | 40.62% | 7.95 pp | -149 | 49 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 760 | 370 | 390 | 48.68% | 48.33% | 48.75% | 1.32 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 760 | 358 | 402 | 47.11% | 43.33% | 48.12% | 2.89 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 760 | 352 | 408 | 46.32% | 45.00% | 46.25% | 3.68 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 760 | 322 | 438 | 42.37% | 35.83% | 40.62% | 7.63 pp | -116 | 44 | -2.64 |
| BTC Daily | rf | RandomForest | 760 | 317 | 443 | 41.71% | 37.92% | 41.88% | 8.29 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 770 | 303 | 467 | 39.35% | 35.83% | 37.08% | 10.65 pp | -164 | 44 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 532 | 259 | 273 | 48.68% | 45.42% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 532 | 255 | 277 | 47.93% | 48.75% | 48.75% | 2.07 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 532 | 253 | 279 | 47.56% | 51.25% | 49.17% | 2.44 pp | -26 | 50 | -0.52 |
| BTC Market Hours | lstm | LSTM | 532 | 230 | 302 | 43.23% | 42.50% | 44.17% | 6.77 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 532 | 229 | 303 | 43.05% | 43.75% | 43.75% | 6.95 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 532 | 221 | 311 | 41.54% | 42.92% | 42.08% | 8.46 pp | -90 | 50 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 586 | 278 | 308 | 47.44% | 50.83% | 48.54% | 2.56 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 586 | 273 | 313 | 46.59% | 46.67% | 47.92% | 3.41 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 586 | 272 | 314 | 46.42% | 51.25% | 47.29% | 3.58 pp | -42 | 50 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 586 | 243 | 343 | 41.47% | 44.17% | 41.25% | 8.53 pp | -100 | 50 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 586 | 238 | 348 | 40.61% | 40.00% | 40.42% | 9.39 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 586 | 233 | 353 | 39.76% | 41.25% | 38.96% | 10.24 pp | -120 | 50 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
