# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T17:25:27.874098+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 330 | 270 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 16:00:00+00:00 | 484 | 258 | 226 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 16:00:00+00:00 | 484 | 258 | 226 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 226 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 23:00:00+00:00 | 226 | 78 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 258 | 136 | 122 | 52.71% | 52.50% | 52.71% | 2.71 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 258 | 129 | 129 | 50.00% | 49.17% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 258 | 127 | 131 | 49.22% | 49.17% | 49.22% | 0.78 pp | -4 | 21 | -0.19 |
| Consolidated Hourly | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 111 | 115 | 49.12% | 49.12% | 49.12% | 0.88 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | nn | NN | 258 | 125 | 133 | 48.45% | 48.33% | 48.45% | 1.55 pp | -8 | 21 | -0.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 235 | 115 | 120 | 48.94% | 48.94% | 48.94% | 1.06 pp | -5 | 10 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 258 | 122 | 136 | 47.29% | 47.50% | 47.29% | 2.71 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 258 | 121 | 137 | 46.90% | 46.67% | 46.90% | 3.10 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 258 | 119 | 139 | 46.12% | 44.58% | 46.12% | 3.88 pp | -20 | 20 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| BTC Market Hours | rf | RandomForest | 258 | 116 | 142 | 44.96% | 43.75% | 44.96% | 5.04 pp | -26 | 20 | -1.30 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 258 | 114 | 144 | 44.19% | 43.75% | 44.19% | 5.81 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 258 | 113 | 145 | 43.80% | 42.50% | 43.80% | 6.20 pp | -32 | 21 | -1.52 |
| BTC Daily | mlp_sklearn | MLPClassifier | 260 | 120 | 140 | 46.15% | 45.83% | 46.15% | 3.85 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 260 | 118 | 142 | 45.38% | 44.58% | 45.38% | 4.62 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 258 | 107 | 151 | 41.47% | 41.67% | 41.47% | 8.53 pp | -44 | 21 | -2.10 |
| Consolidated Hourly | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 98 | 128 | 43.36% | 43.36% | 43.36% | 6.64 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 97 | 129 | 42.92% | 42.92% | 42.92% | 7.08 pp | -32 | 14 | -2.29 |
| BTC Market Hours | lstm | LSTM | 258 | 105 | 153 | 40.70% | 41.67% | 40.70% | 9.30 pp | -48 | 20 | -2.40 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| BTC Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 10 | -2.70 |
| Consolidated Hourly | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 91 | 135 | 40.27% | 40.27% | 40.27% | 9.73 pp | -44 | 14 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Hourly | nn | NN | 235 | 98 | 137 | 41.70% | 41.70% | 41.70% | 8.30 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 260 | 105 | 155 | 40.38% | 39.58% | 40.38% | 9.62 pp | -50 | 12 | -4.17 |
| BTC Hourly | rf | RandomForest | 235 | 96 | 139 | 40.85% | 40.85% | 40.85% | 9.15 pp | -43 | 10 | -4.30 |
| BTC Daily | rf | RandomForest | 260 | 97 | 163 | 37.31% | 37.08% | 37.31% | 12.69 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 270 | 97 | 173 | 35.93% | 35.83% | 35.93% | 14.07 pp | -76 | 13 | -5.85 |
| BTC Hourly | lstm | LSTM | 235 | 87 | 148 | 37.02% | 37.02% | 37.02% | 12.98 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 260 | 92 | 168 | 35.38% | 35.83% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 235 | 80 | 155 | 34.04% | 34.04% | 34.04% | 15.96 pp | -75 | 10 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 235 | 115 | 120 | 48.94% | 48.94% | 48.94% | 1.06 pp | -5 | 10 | -0.50 |
| BTC Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 235 | 98 | 137 | 41.70% | 41.70% | 41.70% | 8.30 pp | -39 | 10 | -3.90 |
| BTC Hourly | rf | RandomForest | 235 | 96 | 139 | 40.85% | 40.85% | 40.85% | 9.15 pp | -43 | 10 | -4.30 |
| BTC Hourly | lstm | LSTM | 235 | 87 | 148 | 37.02% | 37.02% | 37.02% | 12.98 pp | -61 | 10 | -6.10 |
| BTC Hourly | xgb | XGBoost | 235 | 80 | 155 | 34.04% | 34.04% | 34.04% | 15.96 pp | -75 | 10 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 260 | 120 | 140 | 46.15% | 45.83% | 46.15% | 3.85 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 260 | 118 | 142 | 45.38% | 44.58% | 45.38% | 4.62 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 260 | 105 | 155 | 40.38% | 39.58% | 40.38% | 9.62 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 260 | 97 | 163 | 37.31% | 37.08% | 37.31% | 12.69 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 270 | 97 | 173 | 35.93% | 35.83% | 35.93% | 14.07 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 260 | 92 | 168 | 35.38% | 35.83% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 258 | 136 | 122 | 52.71% | 52.50% | 52.71% | 2.71 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 258 | 122 | 136 | 47.29% | 47.50% | 47.29% | 2.71 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 258 | 121 | 137 | 46.90% | 46.67% | 46.90% | 3.10 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 258 | 119 | 139 | 46.12% | 44.58% | 46.12% | 3.88 pp | -20 | 20 | -1.00 |
| BTC Market Hours | rf | RandomForest | 258 | 116 | 142 | 44.96% | 43.75% | 44.96% | 5.04 pp | -26 | 20 | -1.30 |
| BTC Market Hours | lstm | LSTM | 258 | 105 | 153 | 40.70% | 41.67% | 40.70% | 9.30 pp | -48 | 20 | -2.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 258 | 129 | 129 | 50.00% | 49.17% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 258 | 127 | 131 | 49.22% | 49.17% | 49.22% | 0.78 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 258 | 125 | 133 | 48.45% | 48.33% | 48.45% | 1.55 pp | -8 | 21 | -0.38 |
| BTC Market Hours Daily | xgb | XGBoost | 258 | 114 | 144 | 44.19% | 43.75% | 44.19% | 5.81 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 258 | 113 | 145 | 43.80% | 42.50% | 43.80% | 6.20 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 258 | 107 | 151 | 41.47% | 41.67% | 41.47% | 8.53 pp | -44 | 21 | -2.10 |

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
