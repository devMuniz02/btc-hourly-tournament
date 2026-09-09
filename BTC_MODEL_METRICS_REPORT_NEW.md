# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T19:01:36.081279+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 295 | 235 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 331 | 271 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 18:00:00+00:00 | 487 | 259 | 228 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 18:00:00+00:00 | 487 | 259 | 228 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 259 | 137 | 122 | 52.90% | 52.92% | 52.90% | 2.90 pp | 15 | 20 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 259 | 129 | 130 | 49.81% | 49.17% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 259 | 127 | 132 | 49.03% | 48.75% | 49.03% | 0.97 pp | -5 | 21 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 235 | 116 | 119 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | nn | NN | 259 | 125 | 134 | 48.26% | 48.33% | 48.26% | 1.74 pp | -9 | 21 | -0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 259 | 123 | 136 | 47.49% | 47.92% | 47.49% | 2.51 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 259 | 122 | 137 | 47.10% | 47.08% | 47.10% | 2.90 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 259 | 120 | 139 | 46.33% | 44.58% | 46.33% | 3.67 pp | -19 | 20 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Market Hours | rf | RandomForest | 259 | 116 | 143 | 44.79% | 43.33% | 44.79% | 5.21 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 259 | 114 | 145 | 44.02% | 43.33% | 44.02% | 5.98 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 259 | 113 | 146 | 43.63% | 42.50% | 43.63% | 6.37 pp | -33 | 21 | -1.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 261 | 121 | 140 | 46.36% | 45.83% | 46.36% | 3.64 pp | -19 | 12 | -1.58 |
| BTC Daily | nn | NN | 261 | 119 | 142 | 45.59% | 44.58% | 45.59% | 4.41 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 259 | 107 | 152 | 41.31% | 41.67% | 41.31% | 8.69 pp | -45 | 21 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| BTC Market Hours | lstm | LSTM | 259 | 106 | 153 | 40.93% | 42.08% | 40.93% | 9.07 pp | -47 | 20 | -2.35 |
| BTC Hourly | transformer | Transformer | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Hourly | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Hourly | nn | NN | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 10 | -3.70 |
| BTC Daily | transformer | Transformer | 261 | 106 | 155 | 40.61% | 39.58% | 40.61% | 9.39 pp | -49 | 12 | -4.08 |
| BTC Hourly | rf | RandomForest | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 261 | 98 | 163 | 37.55% | 37.08% | 37.55% | 12.45 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 271 | 98 | 173 | 36.16% | 35.83% | 36.16% | 13.84 pp | -75 | 13 | -5.77 |
| BTC Hourly | lstm | LSTM | 235 | 87 | 148 | 37.02% | 37.02% | 37.02% | 12.98 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 261 | 92 | 169 | 35.25% | 35.83% | 35.25% | 14.75 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 235 | 81 | 154 | 34.47% | 34.47% | 34.47% | 15.53 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 235 | 116 | 119 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| BTC Hourly | transformer | Transformer | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 10 | -2.50 |
| BTC Hourly | nn | NN | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 235 | 97 | 138 | 41.28% | 41.28% | 41.28% | 8.72 pp | -41 | 10 | -4.10 |
| BTC Hourly | lstm | LSTM | 235 | 87 | 148 | 37.02% | 37.02% | 37.02% | 12.98 pp | -61 | 10 | -6.10 |
| BTC Hourly | xgb | XGBoost | 235 | 81 | 154 | 34.47% | 34.47% | 34.47% | 15.53 pp | -73 | 10 | -7.30 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 261 | 121 | 140 | 46.36% | 45.83% | 46.36% | 3.64 pp | -19 | 12 | -1.58 |
| BTC Daily | nn | NN | 261 | 119 | 142 | 45.59% | 44.58% | 45.59% | 4.41 pp | -23 | 12 | -1.92 |
| BTC Daily | transformer | Transformer | 261 | 106 | 155 | 40.61% | 39.58% | 40.61% | 9.39 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 261 | 98 | 163 | 37.55% | 37.08% | 37.55% | 12.45 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 271 | 98 | 173 | 36.16% | 35.83% | 36.16% | 13.84 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 261 | 92 | 169 | 35.25% | 35.83% | 35.25% | 14.75 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 259 | 137 | 122 | 52.90% | 52.92% | 52.90% | 2.90 pp | 15 | 20 | 0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 259 | 123 | 136 | 47.49% | 47.92% | 47.49% | 2.51 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 259 | 122 | 137 | 47.10% | 47.08% | 47.10% | 2.90 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 259 | 120 | 139 | 46.33% | 44.58% | 46.33% | 3.67 pp | -19 | 20 | -0.95 |
| BTC Market Hours | rf | RandomForest | 259 | 116 | 143 | 44.79% | 43.33% | 44.79% | 5.21 pp | -27 | 20 | -1.35 |
| BTC Market Hours | lstm | LSTM | 259 | 106 | 153 | 40.93% | 42.08% | 40.93% | 9.07 pp | -47 | 20 | -2.35 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 259 | 129 | 130 | 49.81% | 49.17% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 259 | 127 | 132 | 49.03% | 48.75% | 49.03% | 0.97 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | nn | NN | 259 | 125 | 134 | 48.26% | 48.33% | 48.26% | 1.74 pp | -9 | 21 | -0.43 |
| BTC Market Hours Daily | xgb | XGBoost | 259 | 114 | 145 | 44.02% | 43.33% | 44.02% | 5.98 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 259 | 113 | 146 | 43.63% | 42.50% | 43.63% | 6.37 pp | -33 | 21 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 259 | 107 | 152 | 41.31% | 41.67% | 41.31% | 8.69 pp | -45 | 21 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
