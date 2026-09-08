# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T06:37:15.433736+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 271 | 211 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 307 | 247 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 443 | 235 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 443 | 235 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 203 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T00:00:00+00:00 | 204 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 235 | 121 | 114 | 51.49% | 51.49% | 51.49% | 1.49 pp | 7 | 19 | 0.37 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 211 | 105 | 106 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 235 | 116 | 119 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | transformer | Transformer | 235 | 115 | 120 | 48.94% | 48.94% | 48.94% | 1.06 pp | -5 | 19 | -0.26 |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 235 | 112 | 123 | 47.66% | 47.66% | 47.66% | 2.34 pp | -11 | 19 | -0.58 |
| BTC Market Hours Daily | nn | NN | 235 | 112 | 123 | 47.66% | 47.66% | 47.66% | 2.34 pp | -11 | 19 | -0.58 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| BTC Market Hours | transformer | Transformer | 235 | 110 | 125 | 46.81% | 46.81% | 46.81% | 3.19 pp | -15 | 19 | -0.79 |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | rf | RandomForest | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 19 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 19 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | rf | RandomForest | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 19 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 19 | -1.63 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 235 | 100 | 135 | 42.55% | 42.55% | 42.55% | 7.45 pp | -35 | 19 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 11 | -2.09 |
| BTC Hourly | transformer | Transformer | 211 | 95 | 116 | 45.02% | 45.02% | 45.02% | 4.98 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 235 | 95 | 140 | 40.43% | 40.43% | 40.43% | 9.57 pp | -45 | 19 | -2.37 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
| BTC Hourly | nn | NN | 211 | 89 | 122 | 42.18% | 42.18% | 42.18% | 7.82 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 9 | -3.89 |
| BTC Daily | transformer | Transformer | 237 | 96 | 141 | 40.51% | 40.51% | 40.51% | 9.49 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 237 | 91 | 146 | 38.40% | 38.40% | 38.40% | 11.60 pp | -55 | 11 | -5.00 |
| BTC Hourly | lstm | LSTM | 211 | 79 | 132 | 37.44% | 37.44% | 37.44% | 12.56 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 247 | 88 | 159 | 35.63% | 35.83% | 35.63% | 14.37 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 237 | 80 | 157 | 33.76% | 33.76% | 33.76% | 16.24 pp | -77 | 11 | -7.00 |
| BTC Hourly | xgb | XGBoost | 211 | 73 | 138 | 34.60% | 34.60% | 34.60% | 15.40 pp | -65 | 9 | -7.22 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 211 | 105 | 106 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 211 | 95 | 116 | 45.02% | 45.02% | 45.02% | 4.98 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 211 | 89 | 122 | 42.18% | 42.18% | 42.18% | 7.82 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 9 | -3.89 |
| BTC Hourly | lstm | LSTM | 211 | 79 | 132 | 37.44% | 37.44% | 37.44% | 12.56 pp | -53 | 9 | -5.89 |
| BTC Hourly | xgb | XGBoost | 211 | 73 | 138 | 34.60% | 34.60% | 34.60% | 15.40 pp | -65 | 9 | -7.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 237 | 111 | 126 | 46.84% | 46.84% | 46.84% | 3.16 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 237 | 107 | 130 | 45.15% | 45.15% | 45.15% | 4.85 pp | -23 | 11 | -2.09 |
| BTC Daily | transformer | Transformer | 237 | 96 | 141 | 40.51% | 40.51% | 40.51% | 9.49 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 237 | 91 | 146 | 38.40% | 38.40% | 38.40% | 11.60 pp | -55 | 11 | -5.00 |
| BTC Daily | xgb | XGBoost | 247 | 88 | 159 | 35.63% | 35.83% | 35.63% | 14.37 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 237 | 80 | 157 | 33.76% | 33.76% | 33.76% | 16.24 pp | -77 | 11 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 235 | 121 | 114 | 51.49% | 51.49% | 51.49% | 1.49 pp | 7 | 19 | 0.37 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 235 | 112 | 123 | 47.66% | 47.66% | 47.66% | 2.34 pp | -11 | 19 | -0.58 |
| BTC Market Hours | transformer | Transformer | 235 | 110 | 125 | 46.81% | 46.81% | 46.81% | 3.19 pp | -15 | 19 | -0.79 |
| BTC Market Hours | rf | RandomForest | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 19 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 19 | -1.00 |
| BTC Market Hours | lstm | LSTM | 235 | 100 | 135 | 42.55% | 42.55% | 42.55% | 7.45 pp | -35 | 19 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 235 | 116 | 119 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | transformer | Transformer | 235 | 115 | 120 | 48.94% | 48.94% | 48.94% | 1.06 pp | -5 | 19 | -0.26 |
| BTC Market Hours Daily | nn | NN | 235 | 112 | 123 | 47.66% | 47.66% | 47.66% | 2.34 pp | -11 | 19 | -0.58 |
| BTC Market Hours Daily | rf | RandomForest | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 235 | 102 | 133 | 43.40% | 43.40% | 43.40% | 6.60 pp | -31 | 19 | -1.63 |
| BTC Market Hours Daily | lstm | LSTM | 235 | 95 | 140 | 40.43% | 40.43% | 40.43% | 9.57 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 203 | 93 | 110 | 45.81% | 45.81% | 45.81% | 4.19 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 14 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 203 | 88 | 115 | 43.35% | 43.35% | 43.35% | 6.65 pp | -27 | 14 | -1.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 66 | 26 | 40 | 39.39% | 39.39% | 39.39% | 10.61 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 6 | -2.50 |
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
