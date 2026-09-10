# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T02:19:07.793986+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 300 | 240 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 336 | 276 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 498 | 264 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 498 | 264 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 230 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 230 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 230 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 231 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 264 | 139 | 125 | 52.65% | 52.50% | 52.65% | 2.65 pp | 14 | 21 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 264 | 131 | 133 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 264 | 129 | 135 | 48.86% | 49.17% | 48.86% | 1.14 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 264 | 128 | 136 | 48.48% | 49.17% | 48.48% | 1.52 pp | -8 | 22 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 240 | 117 | 123 | 48.75% | 48.75% | 48.75% | 1.25 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | rf | RandomForest | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 15 | -0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 15 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 264 | 124 | 140 | 46.97% | 47.92% | 46.97% | 3.03 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 264 | 123 | 141 | 46.59% | 46.67% | 46.59% | 3.41 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 264 | 121 | 143 | 45.83% | 44.17% | 45.83% | 4.17 pp | -22 | 21 | -1.05 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 230 | 107 | 123 | 46.52% | 46.52% | 46.52% | 3.48 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 230 | 107 | 123 | 46.52% | 46.52% | 46.52% | 3.48 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 230 | 105 | 125 | 45.65% | 45.65% | 45.65% | 4.35 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 230 | 105 | 125 | 45.65% | 45.65% | 45.65% | 4.35 pp | -20 | 15 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 264 | 116 | 148 | 43.94% | 43.33% | 43.94% | 6.06 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 264 | 116 | 148 | 43.94% | 42.92% | 43.94% | 6.06 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 266 | 123 | 143 | 46.24% | 45.42% | 46.24% | 3.76 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 266 | 123 | 143 | 46.24% | 45.42% | 46.24% | 3.76 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 264 | 113 | 151 | 42.80% | 42.08% | 42.80% | 7.20 pp | -38 | 22 | -1.73 |
| Consolidated Hourly | xgb | XGBoost | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 15 | -1.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 264 | 108 | 156 | 40.91% | 41.67% | 40.91% | 9.09 pp | -48 | 22 | -2.18 |
| Consolidated Hourly | transformer | Transformer | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 15 | -2.27 |
| BTC Market Hours | lstm | LSTM | 264 | 107 | 157 | 40.53% | 42.08% | 40.53% | 9.47 pp | -50 | 21 | -2.38 |
| BTC Hourly | transformer | Transformer | 240 | 108 | 132 | 45.00% | 45.00% | 45.00% | 5.00 pp | -24 | 10 | -2.40 |
| Consolidated Hourly | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 15 | -2.53 |
| Consolidated Daily/Hourly Refresh | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| BTC Hourly | nn | NN | 240 | 101 | 139 | 42.08% | 42.08% | 42.08% | 7.92 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 266 | 108 | 158 | 40.60% | 39.58% | 40.60% | 9.40 pp | -50 | 12 | -4.17 |
| BTC Hourly | rf | RandomForest | 240 | 98 | 142 | 40.83% | 40.83% | 40.83% | 9.17 pp | -44 | 10 | -4.40 |
| BTC Daily | rf | RandomForest | 266 | 101 | 165 | 37.97% | 37.92% | 37.97% | 12.03 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 276 | 102 | 174 | 36.96% | 37.50% | 36.96% | 13.04 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 266 | 95 | 171 | 35.71% | 36.67% | 35.71% | 14.29 pp | -76 | 12 | -6.33 |
| BTC Hourly | lstm | LSTM | 240 | 88 | 152 | 36.67% | 36.67% | 36.67% | 13.33 pp | -64 | 10 | -6.40 |
| BTC Hourly | xgb | XGBoost | 240 | 83 | 157 | 34.58% | 34.58% | 34.58% | 15.42 pp | -74 | 10 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 240 | 117 | 123 | 48.75% | 48.75% | 48.75% | 1.25 pp | -6 | 10 | -0.60 |
| BTC Hourly | transformer | Transformer | 240 | 108 | 132 | 45.00% | 45.00% | 45.00% | 5.00 pp | -24 | 10 | -2.40 |
| BTC Hourly | nn | NN | 240 | 101 | 139 | 42.08% | 42.08% | 42.08% | 7.92 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 240 | 98 | 142 | 40.83% | 40.83% | 40.83% | 9.17 pp | -44 | 10 | -4.40 |
| BTC Hourly | lstm | LSTM | 240 | 88 | 152 | 36.67% | 36.67% | 36.67% | 13.33 pp | -64 | 10 | -6.40 |
| BTC Hourly | xgb | XGBoost | 240 | 83 | 157 | 34.58% | 34.58% | 34.58% | 15.42 pp | -74 | 10 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 266 | 123 | 143 | 46.24% | 45.42% | 46.24% | 3.76 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 266 | 123 | 143 | 46.24% | 45.42% | 46.24% | 3.76 pp | -20 | 12 | -1.67 |
| BTC Daily | transformer | Transformer | 266 | 108 | 158 | 40.60% | 39.58% | 40.60% | 9.40 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 266 | 101 | 165 | 37.97% | 37.92% | 37.97% | 12.03 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 276 | 102 | 174 | 36.96% | 37.50% | 36.96% | 13.04 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 266 | 95 | 171 | 35.71% | 36.67% | 35.71% | 14.29 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 264 | 139 | 125 | 52.65% | 52.50% | 52.65% | 2.65 pp | 14 | 21 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 264 | 124 | 140 | 46.97% | 47.92% | 46.97% | 3.03 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 264 | 123 | 141 | 46.59% | 46.67% | 46.59% | 3.41 pp | -18 | 21 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 264 | 121 | 143 | 45.83% | 44.17% | 45.83% | 4.17 pp | -22 | 21 | -1.05 |
| BTC Market Hours | rf | RandomForest | 264 | 116 | 148 | 43.94% | 42.92% | 43.94% | 6.06 pp | -32 | 21 | -1.52 |
| BTC Market Hours | lstm | LSTM | 264 | 107 | 157 | 40.53% | 42.08% | 40.53% | 9.47 pp | -50 | 21 | -2.38 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 264 | 131 | 133 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 264 | 129 | 135 | 48.86% | 49.17% | 48.86% | 1.14 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 264 | 128 | 136 | 48.48% | 49.17% | 48.48% | 1.52 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | xgb | XGBoost | 264 | 116 | 148 | 43.94% | 43.33% | 43.94% | 6.06 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 264 | 113 | 151 | 42.80% | 42.08% | 42.80% | 7.20 pp | -38 | 22 | -1.73 |
| BTC Market Hours Daily | lstm | LSTM | 264 | 108 | 156 | 40.91% | 41.67% | 40.91% | 9.09 pp | -48 | 22 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 15 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 230 | 107 | 123 | 46.52% | 46.52% | 46.52% | 3.48 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 230 | 105 | 125 | 45.65% | 45.65% | 45.65% | 4.35 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 15 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 15 | -2.27 |
| Consolidated Hourly | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 15 | -2.53 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 15 | -0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 230 | 107 | 123 | 46.52% | 46.52% | 46.52% | 3.48 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 230 | 105 | 125 | 45.65% | 45.65% | 45.65% | 4.35 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 15 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 15 | -2.53 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
