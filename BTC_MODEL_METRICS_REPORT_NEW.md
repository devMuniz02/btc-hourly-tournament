# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T05:03:18.963005+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 302 | 242 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 338 | 278 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 500 | 266 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 500 | 266 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 266 | 140 | 126 | 52.63% | 52.50% | 52.63% | 2.63 pp | 14 | 21 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 266 | 132 | 134 | 49.62% | 48.75% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 266 | 129 | 137 | 48.50% | 48.33% | 48.50% | 1.50 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 266 | 128 | 138 | 48.12% | 48.33% | 48.12% | 1.88 pp | -10 | 22 | -0.45 |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 49.17% | 48.76% | 1.24 pp | -6 | 11 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 266 | 125 | 141 | 46.99% | 47.50% | 46.99% | 3.01 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 266 | 124 | 142 | 46.62% | 46.67% | 46.62% | 3.38 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 266 | 123 | 143 | 46.24% | 45.00% | 46.24% | 3.76 pp | -20 | 21 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| BTC Market Hours Daily | xgb | XGBoost | 266 | 117 | 149 | 43.98% | 42.92% | 43.98% | 6.02 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 266 | 117 | 149 | 43.98% | 42.50% | 43.98% | 6.02 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | rf | RandomForest | 266 | 114 | 152 | 42.86% | 41.67% | 42.86% | 7.14 pp | -38 | 22 | -1.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 266 | 108 | 158 | 40.60% | 41.67% | 40.60% | 9.40 pp | -50 | 22 | -2.27 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| BTC Hourly | transformer | Transformer | 242 | 108 | 134 | 44.63% | 45.00% | 44.63% | 5.37 pp | -26 | 11 | -2.36 |
| BTC Market Hours | lstm | LSTM | 266 | 108 | 158 | 40.60% | 42.50% | 40.60% | 9.40 pp | -50 | 21 | -2.38 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 242 | 102 | 140 | 42.15% | 42.50% | 42.15% | 7.85 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 242 | 99 | 143 | 40.91% | 41.25% | 40.91% | 9.09 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 268 | 109 | 159 | 40.67% | 39.17% | 40.67% | 9.33 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 268 | 102 | 166 | 38.06% | 37.92% | 38.06% | 11.94 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 278 | 103 | 175 | 37.05% | 37.50% | 37.05% | 12.95 pp | -72 | 13 | -5.54 |
| BTC Hourly | lstm | LSTM | 242 | 90 | 152 | 37.19% | 37.50% | 37.19% | 12.81 pp | -62 | 11 | -5.64 |
| BTC Daily | lstm | LSTM | 268 | 96 | 172 | 35.82% | 36.67% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 242 | 84 | 158 | 34.71% | 35.00% | 34.71% | 15.29 pp | -74 | 11 | -6.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 49.17% | 48.76% | 1.24 pp | -6 | 11 | -0.55 |
| BTC Hourly | transformer | Transformer | 242 | 108 | 134 | 44.63% | 45.00% | 44.63% | 5.37 pp | -26 | 11 | -2.36 |
| BTC Hourly | nn | NN | 242 | 102 | 140 | 42.15% | 42.50% | 42.15% | 7.85 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 242 | 99 | 143 | 40.91% | 41.25% | 40.91% | 9.09 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 242 | 90 | 152 | 37.19% | 37.50% | 37.19% | 12.81 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 242 | 84 | 158 | 34.71% | 35.00% | 34.71% | 15.29 pp | -74 | 11 | -6.73 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 268 | 123 | 145 | 45.90% | 45.00% | 45.90% | 4.10 pp | -22 | 12 | -1.83 |
| BTC Daily | transformer | Transformer | 268 | 109 | 159 | 40.67% | 39.17% | 40.67% | 9.33 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 268 | 102 | 166 | 38.06% | 37.92% | 38.06% | 11.94 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 278 | 103 | 175 | 37.05% | 37.50% | 37.05% | 12.95 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 268 | 96 | 172 | 35.82% | 36.67% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 266 | 140 | 126 | 52.63% | 52.50% | 52.63% | 2.63 pp | 14 | 21 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 266 | 125 | 141 | 46.99% | 47.50% | 46.99% | 3.01 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 266 | 124 | 142 | 46.62% | 46.67% | 46.62% | 3.38 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 266 | 123 | 143 | 46.24% | 45.00% | 46.24% | 3.76 pp | -20 | 21 | -0.95 |
| BTC Market Hours | rf | RandomForest | 266 | 117 | 149 | 43.98% | 42.50% | 43.98% | 6.02 pp | -32 | 21 | -1.52 |
| BTC Market Hours | lstm | LSTM | 266 | 108 | 158 | 40.60% | 42.50% | 40.60% | 9.40 pp | -50 | 21 | -2.38 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 266 | 132 | 134 | 49.62% | 48.75% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 266 | 129 | 137 | 48.50% | 48.33% | 48.50% | 1.50 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 266 | 128 | 138 | 48.12% | 48.33% | 48.12% | 1.88 pp | -10 | 22 | -0.45 |
| BTC Market Hours Daily | xgb | XGBoost | 266 | 117 | 149 | 43.98% | 42.92% | 43.98% | 6.02 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 266 | 114 | 152 | 42.86% | 41.67% | 42.86% | 7.14 pp | -38 | 22 | -1.73 |
| BTC Market Hours Daily | lstm | LSTM | 266 | 108 | 158 | 40.60% | 41.67% | 40.60% | 9.40 pp | -50 | 22 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
