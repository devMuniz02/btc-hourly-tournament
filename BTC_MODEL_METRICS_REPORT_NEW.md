# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T14:34:06.964811+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 13:00:00+00:00 | 421 | 224 | 197 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 13:00:00+00:00 | 421 | 224 | 197 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 224 | 116 | 108 | 51.79% | 51.79% | 51.79% | 1.79 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 101 | 99 | 50.50% | 50.50% | 50.50% | 0.50 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 224 | 112 | 112 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 224 | 108 | 116 | 48.21% | 48.21% | 48.21% | 1.79 pp | -8 | 19 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 224 | 106 | 118 | 47.32% | 47.32% | 47.32% | 2.68 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | nn | NN | 224 | 105 | 119 | 46.88% | 46.88% | 46.88% | 3.12 pp | -14 | 19 | -0.74 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 224 | 104 | 120 | 46.43% | 46.43% | 46.43% | 3.57 pp | -16 | 18 | -0.89 |
| BTC Market Hours | rf | RandomForest | 224 | 103 | 121 | 45.98% | 45.98% | 45.98% | 4.02 pp | -18 | 18 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 224 | 100 | 124 | 44.64% | 44.64% | 44.64% | 5.36 pp | -24 | 19 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 224 | 99 | 125 | 44.20% | 44.20% | 44.20% | 5.80 pp | -26 | 18 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 10 | -1.80 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 224 | 94 | 130 | 41.96% | 41.96% | 41.96% | 8.04 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 9 | -2.22 |
| BTC Market Hours | lstm | LSTM | 224 | 92 | 132 | 41.07% | 41.07% | 41.07% | 8.93 pp | -40 | 18 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 224 | 88 | 136 | 39.29% | 39.29% | 39.29% | 10.71 pp | -48 | 19 | -2.53 |
| BTC Daily | nn | NN | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 9 | -3.56 |
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
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 101 | 99 | 50.50% | 50.50% | 50.50% | 0.50 pp | 2 | 9 | 0.22 |
| BTC Hourly | transformer | Transformer | 200 | 90 | 110 | 45.00% | 45.00% | 45.00% | 5.00 pp | -20 | 9 | -2.22 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 9 | -3.56 |
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
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
