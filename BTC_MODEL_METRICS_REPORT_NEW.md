# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T01:50:48.068047+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 284 | 224 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 320 | 260 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 469 | 248 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 469 | 248 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 217 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 248 | 128 | 120 | 51.61% | 52.08% | 51.61% | 1.61 pp | 8 | 20 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 224 | 112 | 112 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 248 | 121 | 127 | 48.79% | 48.33% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 248 | 121 | 127 | 48.79% | 49.17% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 248 | 118 | 130 | 47.58% | 47.92% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | nn | NN | 248 | 118 | 130 | 47.58% | 47.50% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours | xgb | XGBoost | 248 | 117 | 131 | 47.18% | 47.08% | 47.18% | 2.82 pp | -14 | 20 | -0.70 |
| Consolidated Hourly | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| BTC Market Hours | transformer | Transformer | 248 | 116 | 132 | 46.77% | 46.67% | 46.77% | 3.23 pp | -16 | 20 | -0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| BTC Market Hours | rf | RandomForest | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 20 | -1.10 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 248 | 111 | 137 | 44.76% | 44.58% | 44.76% | 5.24 pp | -26 | 20 | -1.30 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 248 | 109 | 139 | 43.95% | 43.33% | 43.95% | 6.05 pp | -30 | 20 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 250 | 116 | 134 | 46.40% | 46.67% | 46.40% | 3.60 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 248 | 105 | 143 | 42.34% | 43.33% | 42.34% | 7.66 pp | -38 | 20 | -1.90 |
| BTC Daily | nn | NN | 250 | 114 | 136 | 45.60% | 45.00% | 45.60% | 4.40 pp | -22 | 11 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 248 | 101 | 147 | 40.73% | 41.67% | 40.73% | 9.27 pp | -46 | 20 | -2.30 |
| BTC Hourly | transformer | Transformer | 224 | 99 | 125 | 44.20% | 44.20% | 44.20% | 5.80 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |
| BTC Hourly | nn | NN | 224 | 94 | 130 | 41.96% | 41.96% | 41.96% | 8.04 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 224 | 93 | 131 | 41.52% | 41.52% | 41.52% | 8.48 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 250 | 100 | 150 | 40.00% | 39.58% | 40.00% | 10.00 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 250 | 95 | 155 | 38.00% | 37.08% | 38.00% | 12.00 pp | -60 | 11 | -5.45 |
| BTC Hourly | lstm | LSTM | 224 | 84 | 140 | 37.50% | 37.50% | 37.50% | 12.50 pp | -56 | 10 | -5.60 |
| BTC Daily | xgb | XGBoost | 260 | 92 | 168 | 35.38% | 35.00% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 224 | 79 | 145 | 35.27% | 35.27% | 35.27% | 14.73 pp | -66 | 10 | -6.60 |
| BTC Daily | lstm | LSTM | 250 | 85 | 165 | 34.00% | 34.58% | 34.00% | 16.00 pp | -80 | 11 | -7.27 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 224 | 112 | 112 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Hourly | transformer | Transformer | 224 | 99 | 125 | 44.20% | 44.20% | 44.20% | 5.80 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 224 | 94 | 130 | 41.96% | 41.96% | 41.96% | 8.04 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 224 | 93 | 131 | 41.52% | 41.52% | 41.52% | 8.48 pp | -38 | 10 | -3.80 |
| BTC Hourly | lstm | LSTM | 224 | 84 | 140 | 37.50% | 37.50% | 37.50% | 12.50 pp | -56 | 10 | -5.60 |
| BTC Hourly | xgb | XGBoost | 224 | 79 | 145 | 35.27% | 35.27% | 35.27% | 14.73 pp | -66 | 10 | -6.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 250 | 116 | 134 | 46.40% | 46.67% | 46.40% | 3.60 pp | -18 | 11 | -1.64 |
| BTC Daily | nn | NN | 250 | 114 | 136 | 45.60% | 45.00% | 45.60% | 4.40 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 250 | 100 | 150 | 40.00% | 39.58% | 40.00% | 10.00 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 250 | 95 | 155 | 38.00% | 37.08% | 38.00% | 12.00 pp | -60 | 11 | -5.45 |
| BTC Daily | xgb | XGBoost | 260 | 92 | 168 | 35.38% | 35.00% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 250 | 85 | 165 | 34.00% | 34.58% | 34.00% | 16.00 pp | -80 | 11 | -7.27 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 248 | 128 | 120 | 51.61% | 52.08% | 51.61% | 1.61 pp | 8 | 20 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 248 | 118 | 130 | 47.58% | 47.92% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours | xgb | XGBoost | 248 | 117 | 131 | 47.18% | 47.08% | 47.18% | 2.82 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 248 | 116 | 132 | 46.77% | 46.67% | 46.77% | 3.23 pp | -16 | 20 | -0.80 |
| BTC Market Hours | rf | RandomForest | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 20 | -1.10 |
| BTC Market Hours | lstm | LSTM | 248 | 105 | 143 | 42.34% | 43.33% | 42.34% | 7.66 pp | -38 | 20 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 248 | 121 | 127 | 48.79% | 48.33% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 248 | 121 | 127 | 48.79% | 49.17% | 48.79% | 1.21 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 248 | 118 | 130 | 47.58% | 47.50% | 47.58% | 2.42 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 248 | 111 | 137 | 44.76% | 44.58% | 44.76% | 5.24 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 248 | 109 | 139 | 43.95% | 43.33% | 43.95% | 6.05 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 248 | 101 | 147 | 40.73% | 41.67% | 40.73% | 9.27 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Hourly | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
