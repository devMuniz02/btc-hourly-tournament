# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T08:07:17.383048+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 304 | 244 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 340 | 280 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 502 | 268 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 502 | 268 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 234 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 234 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 234 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T14:00:00+00:00 | 235 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 268 | 141 | 127 | 52.61% | 52.08% | 52.61% | 2.61 pp | 14 | 21 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 268 | 133 | 135 | 49.63% | 48.75% | 49.63% | 0.37 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 268 | 131 | 137 | 48.88% | 48.75% | 48.88% | 1.12 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 268 | 130 | 138 | 48.51% | 49.17% | 48.51% | 1.49 pp | -8 | 22 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 244 | 118 | 126 | 48.36% | 48.75% | 48.36% | 1.64 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 234 | 111 | 123 | 47.44% | 47.44% | 47.44% | 2.56 pp | -12 | 15 | -0.80 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 234 | 111 | 123 | 47.44% | 47.44% | 47.44% | 2.56 pp | -12 | 15 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 21 | -0.86 |
| BTC Market Hours | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 268 | 124 | 144 | 46.27% | 45.00% | 46.27% | 3.73 pp | -20 | 21 | -0.95 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 15 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 15 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 268 | 119 | 149 | 44.40% | 43.33% | 44.40% | 5.60 pp | -30 | 22 | -1.36 |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 268 | 118 | 150 | 44.03% | 42.08% | 44.03% | 5.97 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 268 | 115 | 153 | 42.91% | 41.67% | 42.91% | 7.09 pp | -38 | 22 | -1.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 270 | 124 | 146 | 45.93% | 44.58% | 45.93% | 4.07 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 270 | 124 | 146 | 45.93% | 45.00% | 45.93% | 4.07 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | xgb | XGBoost | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 234 | 101 | 133 | 43.16% | 43.16% | 43.16% | 6.84 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 234 | 101 | 133 | 43.16% | 43.16% | 43.16% | 6.84 pp | -32 | 15 | -2.13 |
| BTC Hourly | transformer | Transformer | 244 | 110 | 134 | 45.08% | 45.83% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 268 | 110 | 158 | 41.04% | 42.08% | 41.04% | 8.96 pp | -48 | 22 | -2.18 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Market Hours | lstm | LSTM | 268 | 109 | 159 | 40.67% | 42.08% | 40.67% | 9.33 pp | -50 | 21 | -2.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 234 | 97 | 137 | 41.45% | 41.45% | 41.45% | 8.55 pp | -40 | 15 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 234 | 97 | 137 | 41.45% | 41.45% | 41.45% | 8.55 pp | -40 | 15 | -2.67 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Hourly | nn | NN | 244 | 102 | 142 | 41.80% | 42.08% | 41.80% | 8.20 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 244 | 100 | 144 | 40.98% | 41.67% | 40.98% | 9.02 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 270 | 109 | 161 | 40.37% | 38.33% | 40.37% | 9.63 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 270 | 103 | 167 | 38.15% | 37.50% | 38.15% | 11.85 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 280 | 104 | 176 | 37.14% | 37.50% | 37.14% | 12.86 pp | -72 | 13 | -5.54 |
| BTC Hourly | lstm | LSTM | 244 | 90 | 154 | 36.89% | 37.08% | 36.89% | 13.11 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 270 | 96 | 174 | 35.56% | 35.83% | 35.56% | 14.44 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 244 | 85 | 159 | 34.84% | 35.42% | 34.84% | 15.16 pp | -74 | 11 | -6.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 244 | 118 | 126 | 48.36% | 48.75% | 48.36% | 1.64 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 244 | 110 | 134 | 45.08% | 45.83% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 244 | 102 | 142 | 41.80% | 42.08% | 41.80% | 8.20 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 244 | 100 | 144 | 40.98% | 41.67% | 40.98% | 9.02 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 244 | 90 | 154 | 36.89% | 37.08% | 36.89% | 13.11 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 244 | 85 | 159 | 34.84% | 35.42% | 34.84% | 15.16 pp | -74 | 11 | -6.73 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 270 | 124 | 146 | 45.93% | 44.58% | 45.93% | 4.07 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 270 | 124 | 146 | 45.93% | 45.00% | 45.93% | 4.07 pp | -22 | 12 | -1.83 |
| BTC Daily | transformer | Transformer | 270 | 109 | 161 | 40.37% | 38.33% | 40.37% | 9.63 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 270 | 103 | 167 | 38.15% | 37.50% | 38.15% | 11.85 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 280 | 104 | 176 | 37.14% | 37.50% | 37.14% | 12.86 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 270 | 96 | 174 | 35.56% | 35.83% | 35.56% | 14.44 pp | -78 | 12 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 268 | 141 | 127 | 52.61% | 52.08% | 52.61% | 2.61 pp | 14 | 21 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 21 | -0.86 |
| BTC Market Hours | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 268 | 124 | 144 | 46.27% | 45.00% | 46.27% | 3.73 pp | -20 | 21 | -0.95 |
| BTC Market Hours | rf | RandomForest | 268 | 118 | 150 | 44.03% | 42.08% | 44.03% | 5.97 pp | -32 | 21 | -1.52 |
| BTC Market Hours | lstm | LSTM | 268 | 109 | 159 | 40.67% | 42.08% | 40.67% | 9.33 pp | -50 | 21 | -2.38 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 268 | 133 | 135 | 49.63% | 48.75% | 49.63% | 0.37 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 268 | 131 | 137 | 48.88% | 48.75% | 48.88% | 1.12 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 268 | 130 | 138 | 48.51% | 49.17% | 48.51% | 1.49 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | xgb | XGBoost | 268 | 119 | 149 | 44.40% | 43.33% | 44.40% | 5.60 pp | -30 | 22 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 268 | 115 | 153 | 42.91% | 41.67% | 42.91% | 7.09 pp | -38 | 22 | -1.73 |
| BTC Market Hours Daily | lstm | LSTM | 268 | 110 | 158 | 41.04% | 42.08% | 41.04% | 8.96 pp | -48 | 22 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 234 | 111 | 123 | 47.44% | 47.44% | 47.44% | 2.56 pp | -12 | 15 | -0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 234 | 101 | 133 | 43.16% | 43.16% | 43.16% | 6.84 pp | -32 | 15 | -2.13 |
| Consolidated Hourly | nn | NN | 234 | 97 | 137 | 41.45% | 41.45% | 41.45% | 8.55 pp | -40 | 15 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 234 | 111 | 123 | 47.44% | 47.44% | 47.44% | 2.56 pp | -12 | 15 | -0.80 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 234 | 108 | 126 | 46.15% | 46.15% | 46.15% | 3.85 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 234 | 107 | 127 | 45.73% | 45.73% | 45.73% | 4.27 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 15 | -1.87 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 234 | 101 | 133 | 43.16% | 43.16% | 43.16% | 6.84 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 234 | 97 | 137 | 41.45% | 41.45% | 41.45% | 8.55 pp | -40 | 15 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
