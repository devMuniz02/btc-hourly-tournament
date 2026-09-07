# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T13:43:52.134469+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 260 | 200 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 296 | 236 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 420 | 224 | 196 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 420 | 224 | 196 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 194 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T18:00:00+00:00 | 195 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 224 | 116 | 108 | 51.79% | 51.79% | 51.79% | 1.79 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | transformer | Transformer | 224 | 112 | 112 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 100 | 100 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 224 | 108 | 116 | 48.21% | 48.21% | 48.21% | 1.79 pp | -8 | 19 | -0.42 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 224 | 106 | 118 | 47.32% | 47.32% | 47.32% | 2.68 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | nn | NN | 224 | 105 | 119 | 46.88% | 46.88% | 46.88% | 3.12 pp | -14 | 19 | -0.74 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 224 | 104 | 120 | 46.43% | 46.43% | 46.43% | 3.57 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 19 | -1.26 |
| Consolidated Hourly | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 224 | 99 | 125 | 44.20% | 44.20% | 44.20% | 5.80 pp | -26 | 18 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 10 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 224 | 94 | 130 | 41.96% | 41.96% | 41.96% | 8.04 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 224 | 92 | 132 | 41.07% | 41.07% | 41.07% | 8.93 pp | -40 | 18 | -2.22 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 200 | 89 | 111 | 44.50% | 44.50% | 44.50% | 5.50 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | lstm | LSTM | 224 | 88 | 136 | 39.29% | 39.29% | 39.29% | 10.71 pp | -48 | 19 | -2.53 |
| BTC Daily | nn | NN | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 83 | 117 | 41.50% | 41.50% | 41.50% | 8.50 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 10 | -4.40 |
| BTC Hourly | lstm | LSTM | 200 | 76 | 124 | 38.00% | 38.00% | 38.00% | 12.00 pp | -48 | 9 | -5.33 |
| BTC Daily | rf | RandomForest | 226 | 86 | 140 | 38.05% | 38.05% | 38.05% | 11.95 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 236 | 83 | 153 | 35.17% | 35.17% | 35.17% | 14.83 pp | -70 | 11 | -6.36 |
| BTC Hourly | xgb | XGBoost | 200 | 71 | 129 | 35.50% | 35.50% | 35.50% | 14.50 pp | -58 | 9 | -6.44 |
| BTC Daily | lstm | LSTM | 226 | 75 | 151 | 33.19% | 33.19% | 33.19% | 16.81 pp | -76 | 10 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 100 | 100 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 200 | 89 | 111 | 44.50% | 44.50% | 44.50% | 5.50 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 83 | 117 | 41.50% | 41.50% | 41.50% | 8.50 pp | -34 | 9 | -3.78 |
| BTC Hourly | lstm | LSTM | 200 | 76 | 124 | 38.00% | 38.00% | 38.00% | 12.00 pp | -48 | 9 | -5.33 |
| BTC Hourly | xgb | XGBoost | 200 | 71 | 129 | 35.50% | 35.50% | 35.50% | 14.50 pp | -58 | 9 | -6.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 10 | -1.80 |
| BTC Daily | nn | NN | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 10 | -2.60 |
| BTC Daily | transformer | Transformer | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 10 | -4.40 |
| BTC Daily | rf | RandomForest | 226 | 86 | 140 | 38.05% | 38.05% | 38.05% | 11.95 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 236 | 83 | 153 | 35.17% | 35.17% | 35.17% | 14.83 pp | -70 | 11 | -6.36 |
| BTC Daily | lstm | LSTM | 226 | 75 | 151 | 33.19% | 33.19% | 33.19% | 16.81 pp | -76 | 10 | -7.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 224 | 116 | 108 | 51.79% | 51.79% | 51.79% | 1.79 pp | 8 | 18 | 0.44 |
| BTC Market Hours | transformer | Transformer | 224 | 106 | 118 | 47.32% | 47.32% | 47.32% | 2.68 pp | -12 | 18 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 224 | 104 | 120 | 46.43% | 46.43% | 46.43% | 3.57 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 224 | 99 | 125 | 44.20% | 44.20% | 44.20% | 5.80 pp | -26 | 18 | -1.44 |
| BTC Market Hours | lstm | LSTM | 224 | 92 | 132 | 41.07% | 41.07% | 41.07% | 8.93 pp | -40 | 18 | -2.22 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 224 | 112 | 112 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 224 | 108 | 116 | 48.21% | 48.21% | 48.21% | 1.79 pp | -8 | 19 | -0.42 |
| BTC Market Hours Daily | nn | NN | 224 | 105 | 119 | 46.88% | 46.88% | 46.88% | 3.12 pp | -14 | 19 | -0.74 |
| BTC Market Hours Daily | rf | RandomForest | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 19 | -1.26 |
| BTC Market Hours Daily | xgb | XGBoost | 224 | 94 | 130 | 41.96% | 41.96% | 41.96% | 8.04 pp | -36 | 19 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 224 | 88 | 136 | 39.29% | 39.29% | 39.29% | 10.71 pp | -48 | 19 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 194 | 95 | 99 | 48.97% | 48.97% | 48.97% | 1.03 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 194 | 88 | 106 | 45.36% | 45.36% | 45.36% | 4.64 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 13 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
