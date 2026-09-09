# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T23:55:15.936411+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 298 | 238 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 334 | 274 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 22:00:00+00:00 | 494 | 262 | 232 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 22:00:00+00:00 | 494 | 262 | 232 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 229 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 12:00:00+00:00 | 229 | 80 | 149 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 262 | 138 | 124 | 52.67% | 52.50% | 52.67% | 2.67 pp | 14 | 21 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 262 | 130 | 132 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 262 | 129 | 133 | 49.24% | 49.17% | 49.24% | 0.76 pp | -4 | 22 | -0.18 |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 238 | 117 | 121 | 49.16% | 49.16% | 49.16% | 0.84 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | nn | NN | 262 | 126 | 136 | 48.09% | 48.33% | 48.09% | 1.91 pp | -10 | 22 | -0.45 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 262 | 123 | 139 | 46.95% | 47.92% | 46.95% | 3.05 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 262 | 123 | 139 | 46.95% | 47.08% | 46.95% | 3.05 pp | -16 | 21 | -0.76 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 262 | 120 | 142 | 45.80% | 44.17% | 45.80% | 4.20 pp | -22 | 21 | -1.05 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| BTC Market Hours | rf | RandomForest | 262 | 116 | 146 | 44.27% | 43.33% | 44.27% | 5.73 pp | -30 | 21 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 262 | 115 | 147 | 43.89% | 42.92% | 43.89% | 6.11 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 262 | 113 | 149 | 43.13% | 42.08% | 43.13% | 6.87 pp | -36 | 22 | -1.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 264 | 122 | 142 | 46.21% | 45.42% | 46.21% | 3.79 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 264 | 122 | 142 | 46.21% | 45.42% | 46.21% | 3.79 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 262 | 108 | 154 | 41.22% | 42.08% | 41.22% | 8.78 pp | -46 | 22 | -2.09 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| BTC Market Hours | lstm | LSTM | 262 | 107 | 155 | 40.84% | 42.50% | 40.84% | 9.16 pp | -48 | 21 | -2.29 |
| BTC Hourly | transformer | Transformer | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| BTC Hourly | nn | NN | 238 | 101 | 137 | 42.44% | 42.44% | 42.44% | 7.56 pp | -36 | 10 | -3.60 |
| BTC Daily | transformer | Transformer | 264 | 108 | 156 | 40.91% | 40.00% | 40.91% | 9.09 pp | -48 | 12 | -4.00 |
| BTC Hourly | rf | RandomForest | 238 | 98 | 140 | 41.18% | 41.18% | 41.18% | 8.82 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 264 | 99 | 165 | 37.50% | 37.08% | 37.50% | 12.50 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 274 | 100 | 174 | 36.50% | 36.67% | 36.50% | 13.50 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 238 | 88 | 150 | 36.97% | 36.97% | 36.97% | 13.03 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 264 | 93 | 171 | 35.23% | 35.83% | 35.23% | 14.77 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 238 | 82 | 156 | 34.45% | 34.45% | 34.45% | 15.55 pp | -74 | 10 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 238 | 117 | 121 | 49.16% | 49.16% | 49.16% | 0.84 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 238 | 107 | 131 | 44.96% | 44.96% | 44.96% | 5.04 pp | -24 | 10 | -2.40 |
| BTC Hourly | nn | NN | 238 | 101 | 137 | 42.44% | 42.44% | 42.44% | 7.56 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 238 | 98 | 140 | 41.18% | 41.18% | 41.18% | 8.82 pp | -42 | 10 | -4.20 |
| BTC Hourly | lstm | LSTM | 238 | 88 | 150 | 36.97% | 36.97% | 36.97% | 13.03 pp | -62 | 10 | -6.20 |
| BTC Hourly | xgb | XGBoost | 238 | 82 | 156 | 34.45% | 34.45% | 34.45% | 15.55 pp | -74 | 10 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 264 | 122 | 142 | 46.21% | 45.42% | 46.21% | 3.79 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 264 | 122 | 142 | 46.21% | 45.42% | 46.21% | 3.79 pp | -20 | 12 | -1.67 |
| BTC Daily | transformer | Transformer | 264 | 108 | 156 | 40.91% | 40.00% | 40.91% | 9.09 pp | -48 | 12 | -4.00 |
| BTC Daily | rf | RandomForest | 264 | 99 | 165 | 37.50% | 37.08% | 37.50% | 12.50 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 274 | 100 | 174 | 36.50% | 36.67% | 36.50% | 13.50 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 264 | 93 | 171 | 35.23% | 35.83% | 35.23% | 14.77 pp | -78 | 12 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 262 | 138 | 124 | 52.67% | 52.50% | 52.67% | 2.67 pp | 14 | 21 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 262 | 123 | 139 | 46.95% | 47.92% | 46.95% | 3.05 pp | -16 | 21 | -0.76 |
| BTC Market Hours | transformer | Transformer | 262 | 123 | 139 | 46.95% | 47.08% | 46.95% | 3.05 pp | -16 | 21 | -0.76 |
| BTC Market Hours | xgb | XGBoost | 262 | 120 | 142 | 45.80% | 44.17% | 45.80% | 4.20 pp | -22 | 21 | -1.05 |
| BTC Market Hours | rf | RandomForest | 262 | 116 | 146 | 44.27% | 43.33% | 44.27% | 5.73 pp | -30 | 21 | -1.43 |
| BTC Market Hours | lstm | LSTM | 262 | 107 | 155 | 40.84% | 42.50% | 40.84% | 9.16 pp | -48 | 21 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 262 | 130 | 132 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 262 | 129 | 133 | 49.24% | 49.17% | 49.24% | 0.76 pp | -4 | 22 | -0.18 |
| BTC Market Hours Daily | nn | NN | 262 | 126 | 136 | 48.09% | 48.33% | 48.09% | 1.91 pp | -10 | 22 | -0.45 |
| BTC Market Hours Daily | xgb | XGBoost | 262 | 115 | 147 | 43.89% | 42.92% | 43.89% | 6.11 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 262 | 113 | 149 | 43.13% | 42.08% | 43.13% | 6.87 pp | -36 | 22 | -1.64 |
| BTC Market Hours Daily | lstm | LSTM | 262 | 108 | 154 | 41.22% | 42.08% | 41.22% | 8.78 pp | -46 | 22 | -2.09 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 107 | 122 | 46.72% | 46.72% | 46.72% | 3.28 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 99 | 130 | 43.23% | 43.23% | 43.23% | 6.77 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 98 | 131 | 42.79% | 42.79% | 42.79% | 7.21 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 15 | -3.00 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
