# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T10:19:05.535903+00:00
Scope: `new`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 192 | 132 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 228 | 168 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 299 | 156 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 299 | 156 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 131 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 131 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 27 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 27 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 12 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Market Hours | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 158 | 79 | 79 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 156 | 77 | 79 | 49.36% | 49.36% | 49.36% | 0.64 pp | -2 | 13 | -0.15 |
| Consolidated Market Hours | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 156 | 74 | 82 | 47.44% | 47.44% | 47.44% | 2.56 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 13 | -1.08 |
| BTC Daily | nn | NN | 158 | 75 | 83 | 47.47% | 47.47% | 47.47% | 2.53 pp | -8 | 7 | -1.14 |
| BTC Market Hours | rf | RandomForest | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 156 | 70 | 86 | 44.87% | 44.87% | 44.87% | 5.13 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 13 | -1.69 |
| BTC Hourly | nn | NN | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 7 | -2.29 |
| BTC Market Hours | lstm | LSTM | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 12 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 12 | -2.33 |
| BTC Hourly | rf | RandomForest | 132 | 57 | 75 | 43.18% | 43.18% | 43.18% | 6.82 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 156 | 58 | 98 | 37.18% | 37.18% | 37.18% | 12.82 pp | -40 | 13 | -3.08 |
| BTC Daily | rf | RandomForest | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| BTC Hourly | xgb | XGBoost | 132 | 50 | 82 | 37.88% | 37.88% | 37.88% | 12.12 pp | -32 | 6 | -5.33 |
| BTC Daily | lstm | LSTM | 158 | 60 | 98 | 37.97% | 37.97% | 37.97% | 12.03 pp | -38 | 7 | -5.43 |
| BTC Daily | xgb | XGBoost | 168 | 62 | 106 | 36.90% | 36.90% | 36.90% | 13.10 pp | -44 | 8 | -5.50 |
| BTC Hourly | lstm | LSTM | 132 | 48 | 84 | 36.36% | 36.36% | 36.36% | 13.64 pp | -36 | 6 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | nn | NN | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 6 | -2.00 |
| BTC Hourly | rf | RandomForest | 132 | 57 | 75 | 43.18% | 43.18% | 43.18% | 6.82 pp | -18 | 6 | -3.00 |
| BTC Hourly | xgb | XGBoost | 132 | 50 | 82 | 37.88% | 37.88% | 37.88% | 12.12 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 132 | 48 | 84 | 36.36% | 36.36% | 36.36% | 13.64 pp | -36 | 6 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 158 | 79 | 79 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Daily | nn | NN | 158 | 75 | 83 | 47.47% | 47.47% | 47.47% | 2.53 pp | -8 | 7 | -1.14 |
| BTC Daily | transformer | Transformer | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 7 | -3.14 |
| BTC Daily | lstm | LSTM | 158 | 60 | 98 | 37.97% | 37.97% | 37.97% | 12.03 pp | -38 | 7 | -5.43 |
| BTC Daily | xgb | XGBoost | 168 | 62 | 106 | 36.90% | 36.90% | 36.90% | 13.10 pp | -44 | 8 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 12 | 0.67 |
| BTC Market Hours | rf | RandomForest | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 12 | -1.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 156 | 70 | 86 | 44.87% | 44.87% | 44.87% | 5.13 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 12 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 12 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 156 | 77 | 79 | 49.36% | 49.36% | 49.36% | 0.64 pp | -2 | 13 | -0.15 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 156 | 74 | 82 | 47.44% | 47.44% | 47.44% | 2.56 pp | -8 | 13 | -0.62 |
| BTC Market Hours Daily | nn | NN | 156 | 71 | 85 | 45.51% | 45.51% | 45.51% | 4.49 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 13 | -1.69 |
| BTC Market Hours Daily | xgb | XGBoost | 156 | 64 | 92 | 41.03% | 41.03% | 41.03% | 8.97 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 156 | 58 | 98 | 37.18% | 37.18% | 37.18% | 12.82 pp | -40 | 13 | -3.08 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
