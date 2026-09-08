# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T08:34:42.972293+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 273 | 213 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 308 | 248 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 444 | 236 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 444 | 236 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 205 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 12:00:00+00:00 | 205 | 67 | 138 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 236 | 121 | 115 | 51.27% | 51.27% | 51.27% | 1.27 pp | 6 | 19 | 0.32 |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 213 | 105 | 108 | 49.30% | 49.30% | 49.30% | 0.70 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | nn | NN | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 19 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 20 | -1.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 11 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 236 | 103 | 133 | 43.64% | 43.64% | 43.64% | 6.36 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 236 | 101 | 135 | 42.80% | 42.80% | 42.80% | 7.20 pp | -34 | 19 | -1.79 |
| BTC Daily | nn | NN | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 11 | -2.00 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 20 | -2.20 |
| BTC Hourly | transformer | Transformer | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 9 | -3.89 |
| BTC Hourly | rf | RandomForest | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 9 | -4.11 |
| BTC Daily | transformer | Transformer | 238 | 96 | 142 | 40.34% | 40.34% | 40.34% | 9.66 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 238 | 91 | 147 | 38.24% | 38.24% | 38.24% | 11.76 pp | -56 | 11 | -5.09 |
| BTC Daily | xgb | XGBoost | 248 | 89 | 159 | 35.89% | 36.25% | 35.89% | 14.11 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 213 | 79 | 134 | 37.09% | 37.09% | 37.09% | 12.91 pp | -55 | 9 | -6.11 |
| BTC Daily | lstm | LSTM | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 11 | -6.91 |
| BTC Hourly | xgb | XGBoost | 213 | 73 | 140 | 34.27% | 34.27% | 34.27% | 15.73 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 213 | 105 | 108 | 49.30% | 49.30% | 49.30% | 0.70 pp | -3 | 9 | -0.33 |
| BTC Hourly | transformer | Transformer | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 9 | -3.89 |
| BTC Hourly | rf | RandomForest | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 213 | 79 | 134 | 37.09% | 37.09% | 37.09% | 12.91 pp | -55 | 9 | -6.11 |
| BTC Hourly | xgb | XGBoost | 213 | 73 | 140 | 34.27% | 34.27% | 34.27% | 15.73 pp | -67 | 9 | -7.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 238 | 111 | 127 | 46.64% | 46.64% | 46.64% | 3.36 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 238 | 96 | 142 | 40.34% | 40.34% | 40.34% | 9.66 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 238 | 91 | 147 | 38.24% | 38.24% | 38.24% | 11.76 pp | -56 | 11 | -5.09 |
| BTC Daily | xgb | XGBoost | 248 | 89 | 159 | 35.89% | 36.25% | 35.89% | 14.11 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 11 | -6.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 236 | 121 | 115 | 51.27% | 51.27% | 51.27% | 1.27 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 236 | 110 | 126 | 46.61% | 46.61% | 46.61% | 3.39 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 236 | 108 | 128 | 45.76% | 45.76% | 45.76% | 4.24 pp | -20 | 19 | -1.05 |
| BTC Market Hours | lstm | LSTM | 236 | 101 | 135 | 42.80% | 42.80% | 42.80% | 7.20 pp | -34 | 19 | -1.79 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | transformer | Transformer | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | nn | NN | 236 | 112 | 124 | 47.46% | 47.46% | 47.46% | 2.54 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | rf | RandomForest | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 236 | 103 | 133 | 43.64% | 43.64% | 43.64% | 6.36 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 236 | 96 | 140 | 40.68% | 40.68% | 40.68% | 9.32 pp | -44 | 20 | -2.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 14 | 0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 205 | 95 | 110 | 46.34% | 46.34% | 46.34% | 3.66 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 205 | 91 | 114 | 44.39% | 44.39% | 44.39% | 5.61 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 205 | 90 | 115 | 43.90% | 43.90% | 43.90% | 6.10 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 205 | 88 | 117 | 42.93% | 42.93% | 42.93% | 7.07 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
