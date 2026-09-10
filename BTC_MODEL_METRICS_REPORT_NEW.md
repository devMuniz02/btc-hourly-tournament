# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T03:21:29.895648+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 301 | 241 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 337 | 277 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 499 | 265 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 498 | 264 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 231 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 13:00:00+00:00 | 231 | 81 | 150 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 265 | 139 | 126 | 52.45% | 52.08% | 52.45% | 2.45 pp | 13 | 21 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 264 | 131 | 133 | 49.62% | 49.17% | 49.62% | 0.38 pp | -2 | 22 | -0.09 |
| BTC Market Hours Daily | transformer | Transformer | 264 | 129 | 135 | 48.86% | 49.17% | 48.86% | 1.14 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 264 | 128 | 136 | 48.48% | 49.17% | 48.48% | 1.52 pp | -8 | 22 | -0.36 |
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 241 | 117 | 124 | 48.55% | 48.75% | 48.55% | 1.45 pp | -7 | 10 | -0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 265 | 124 | 141 | 46.79% | 47.50% | 46.79% | 3.21 pp | -17 | 21 | -0.81 |
| BTC Market Hours | transformer | Transformer | 265 | 123 | 142 | 46.42% | 46.67% | 46.42% | 3.58 pp | -19 | 21 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 265 | 122 | 143 | 46.04% | 44.58% | 46.04% | 3.96 pp | -21 | 21 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| BTC Market Hours Daily | xgb | XGBoost | 264 | 116 | 148 | 43.94% | 43.33% | 43.94% | 6.06 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 265 | 117 | 148 | 44.15% | 42.92% | 44.15% | 5.85 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | rf | RandomForest | 264 | 113 | 151 | 42.80% | 42.08% | 42.80% | 7.20 pp | -38 | 22 | -1.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 267 | 123 | 144 | 46.07% | 45.42% | 46.07% | 3.93 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 267 | 123 | 144 | 46.07% | 45.42% | 46.07% | 3.93 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 264 | 108 | 156 | 40.91% | 41.67% | 40.91% | 9.09 pp | -48 | 22 | -2.18 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| BTC Market Hours | lstm | LSTM | 265 | 107 | 158 | 40.38% | 42.08% | 40.38% | 9.62 pp | -51 | 21 | -2.43 |
| BTC Hourly | transformer | Transformer | 241 | 108 | 133 | 44.81% | 45.00% | 44.81% | 5.19 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 241 | 102 | 139 | 42.32% | 42.50% | 42.32% | 7.68 pp | -37 | 10 | -3.70 |
| BTC Daily | transformer | Transformer | 267 | 109 | 158 | 40.82% | 39.58% | 40.82% | 9.18 pp | -49 | 12 | -4.08 |
| BTC Hourly | rf | RandomForest | 241 | 99 | 142 | 41.08% | 41.25% | 41.08% | 8.92 pp | -43 | 10 | -4.30 |
| BTC Daily | rf | RandomForest | 267 | 101 | 166 | 37.83% | 37.92% | 37.83% | 12.17 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 277 | 102 | 175 | 36.82% | 37.50% | 36.82% | 13.18 pp | -73 | 13 | -5.62 |
| BTC Hourly | lstm | LSTM | 241 | 89 | 152 | 36.93% | 37.08% | 36.93% | 13.07 pp | -63 | 10 | -6.30 |
| BTC Daily | lstm | LSTM | 267 | 95 | 172 | 35.58% | 36.67% | 35.58% | 14.42 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 241 | 84 | 157 | 34.85% | 35.00% | 34.85% | 15.15 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 241 | 117 | 124 | 48.55% | 48.75% | 48.55% | 1.45 pp | -7 | 10 | -0.70 |
| BTC Hourly | transformer | Transformer | 241 | 108 | 133 | 44.81% | 45.00% | 44.81% | 5.19 pp | -25 | 10 | -2.50 |
| BTC Hourly | nn | NN | 241 | 102 | 139 | 42.32% | 42.50% | 42.32% | 7.68 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 241 | 99 | 142 | 41.08% | 41.25% | 41.08% | 8.92 pp | -43 | 10 | -4.30 |
| BTC Hourly | lstm | LSTM | 241 | 89 | 152 | 36.93% | 37.08% | 36.93% | 13.07 pp | -63 | 10 | -6.30 |
| BTC Hourly | xgb | XGBoost | 241 | 84 | 157 | 34.85% | 35.00% | 34.85% | 15.15 pp | -73 | 10 | -7.30 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 267 | 123 | 144 | 46.07% | 45.42% | 46.07% | 3.93 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 267 | 123 | 144 | 46.07% | 45.42% | 46.07% | 3.93 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 267 | 109 | 158 | 40.82% | 39.58% | 40.82% | 9.18 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 267 | 101 | 166 | 37.83% | 37.92% | 37.83% | 12.17 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 277 | 102 | 175 | 36.82% | 37.50% | 36.82% | 13.18 pp | -73 | 13 | -5.62 |
| BTC Daily | lstm | LSTM | 267 | 95 | 172 | 35.58% | 36.67% | 35.58% | 14.42 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 265 | 139 | 126 | 52.45% | 52.08% | 52.45% | 2.45 pp | 13 | 21 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 265 | 124 | 141 | 46.79% | 47.50% | 46.79% | 3.21 pp | -17 | 21 | -0.81 |
| BTC Market Hours | transformer | Transformer | 265 | 123 | 142 | 46.42% | 46.67% | 46.42% | 3.58 pp | -19 | 21 | -0.90 |
| BTC Market Hours | xgb | XGBoost | 265 | 122 | 143 | 46.04% | 44.58% | 46.04% | 3.96 pp | -21 | 21 | -1.00 |
| BTC Market Hours | rf | RandomForest | 265 | 117 | 148 | 44.15% | 42.92% | 44.15% | 5.85 pp | -31 | 21 | -1.48 |
| BTC Market Hours | lstm | LSTM | 265 | 107 | 158 | 40.38% | 42.08% | 40.38% | 9.62 pp | -51 | 21 | -2.43 |

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
| Consolidated Hourly | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 231 | 112 | 119 | 48.48% | 48.48% | 48.48% | 1.52 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 231 | 108 | 123 | 46.75% | 46.75% | 46.75% | 3.25 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 231 | 98 | 133 | 42.42% | 42.42% | 42.42% | 7.58 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
