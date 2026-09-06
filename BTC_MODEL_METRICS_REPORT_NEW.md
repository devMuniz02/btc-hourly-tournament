# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T10:29:13.955030+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 241 | 181 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 277 | 217 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 387 | 205 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 387 | 205 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 177 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 177 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 51 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 22:00:00+00:00 | 177 | 51 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 205 | 106 | 99 | 51.71% | 51.71% | 51.71% | 1.71 pp | 7 | 16 | 0.44 |
| BTC Market Hours Daily | transformer | Transformer | 205 | 106 | 99 | 51.71% | 51.71% | 51.71% | 1.71 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 181 | 92 | 89 | 50.83% | 50.83% | 50.83% | 0.83 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 205 | 100 | 105 | 48.78% | 48.78% | 48.78% | 1.22 pp | -5 | 17 | -0.29 |
| BTC Market Hours | transformer | Transformer | 205 | 100 | 105 | 48.78% | 48.78% | 48.78% | 1.22 pp | -5 | 16 | -0.31 |
| BTC Market Hours Daily | nn | NN | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 17 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 205 | 96 | 109 | 46.83% | 46.83% | 46.83% | 3.17 pp | -13 | 16 | -0.81 |
| BTC Daily | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 12 | -1.08 |
| BTC Market Hours | rf | RandomForest | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 16 | -1.19 |
| Consolidated Market Hours | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Hourly | transformer | Transformer | 181 | 85 | 96 | 46.96% | 46.96% | 46.96% | 3.04 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 17 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 16 | -1.69 |
| Consolidated Hourly | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 205 | 86 | 119 | 41.95% | 41.95% | 41.95% | 8.05 pp | -33 | 16 | -2.06 |
| Consolidated Hourly | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 205 | 83 | 122 | 40.49% | 40.49% | 40.49% | 9.51 pp | -39 | 17 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 205 | 82 | 123 | 40.00% | 40.00% | 40.00% | 10.00 pp | -41 | 17 | -2.41 |
| BTC Daily | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 9 | -2.56 |
| Consolidated Market Hours | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 207 | 86 | 121 | 41.55% | 41.55% | 41.55% | 8.45 pp | -35 | 9 | -3.89 |
| BTC Hourly | xgb | XGBoost | 181 | 68 | 113 | 37.57% | 37.57% | 37.57% | 12.43 pp | -45 | 8 | -5.62 |
| BTC Hourly | lstm | LSTM | 181 | 67 | 114 | 37.02% | 37.02% | 37.02% | 12.98 pp | -47 | 8 | -5.88 |
| BTC Daily | rf | RandomForest | 207 | 77 | 130 | 37.20% | 37.20% | 37.20% | 12.80 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 217 | 78 | 139 | 35.94% | 35.94% | 35.94% | 14.06 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 207 | 69 | 138 | 33.33% | 33.33% | 33.33% | 16.67 pp | -69 | 9 | -7.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 181 | 92 | 89 | 50.83% | 50.83% | 50.83% | 0.83 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 181 | 85 | 96 | 46.96% | 46.96% | 46.96% | 3.04 pp | -11 | 8 | -1.38 |
| BTC Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 181 | 68 | 113 | 37.57% | 37.57% | 37.57% | 12.43 pp | -45 | 8 | -5.62 |
| BTC Hourly | lstm | LSTM | 181 | 67 | 114 | 37.02% | 37.02% | 37.02% | 12.98 pp | -47 | 8 | -5.88 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 9 | -1.00 |
| BTC Daily | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 207 | 86 | 121 | 41.55% | 41.55% | 41.55% | 8.45 pp | -35 | 9 | -3.89 |
| BTC Daily | rf | RandomForest | 207 | 77 | 130 | 37.20% | 37.20% | 37.20% | 12.80 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 217 | 78 | 139 | 35.94% | 35.94% | 35.94% | 14.06 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 207 | 69 | 138 | 33.33% | 33.33% | 33.33% | 16.67 pp | -69 | 9 | -7.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 205 | 106 | 99 | 51.71% | 51.71% | 51.71% | 1.71 pp | 7 | 16 | 0.44 |
| BTC Market Hours | transformer | Transformer | 205 | 100 | 105 | 48.78% | 48.78% | 48.78% | 1.22 pp | -5 | 16 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 205 | 96 | 109 | 46.83% | 46.83% | 46.83% | 3.17 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 205 | 93 | 112 | 45.37% | 45.37% | 45.37% | 4.63 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 205 | 89 | 116 | 43.41% | 43.41% | 43.41% | 6.59 pp | -27 | 16 | -1.69 |
| BTC Market Hours | lstm | LSTM | 205 | 86 | 119 | 41.95% | 41.95% | 41.95% | 8.05 pp | -33 | 16 | -2.06 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 205 | 106 | 99 | 51.71% | 51.71% | 51.71% | 1.71 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 205 | 100 | 105 | 48.78% | 48.78% | 48.78% | 1.22 pp | -5 | 17 | -0.29 |
| BTC Market Hours Daily | nn | NN | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 205 | 83 | 122 | 40.49% | 40.49% | 40.49% | 9.51 pp | -39 | 17 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 205 | 82 | 123 | 40.00% | 40.00% | 40.00% | 10.00 pp | -41 | 17 | -2.41 |

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
