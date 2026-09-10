# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T07:36:46.548835+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 339 | 279 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 501 | 267 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 501 | 267 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 233 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 14:00:00+00:00 | 233 | 82 | 151 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 267 | 141 | 126 | 52.81% | 52.50% | 52.81% | 2.81 pp | 15 | 21 | 0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 267 | 132 | 135 | 49.44% | 48.75% | 49.44% | 0.56 pp | -3 | 22 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 267 | 130 | 137 | 48.69% | 48.33% | 48.69% | 1.31 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | nn | NN | 267 | 129 | 138 | 48.31% | 48.75% | 48.31% | 1.69 pp | -9 | 22 | -0.41 |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 244 | 118 | 126 | 48.36% | 48.75% | 48.36% | 1.64 pp | -8 | 11 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 267 | 125 | 142 | 46.82% | 47.08% | 46.82% | 3.18 pp | -17 | 21 | -0.81 |
| BTC Market Hours | transformer | Transformer | 267 | 125 | 142 | 46.82% | 46.67% | 46.82% | 3.18 pp | -17 | 21 | -0.81 |
| BTC Market Hours | xgb | XGBoost | 267 | 124 | 143 | 46.44% | 45.42% | 46.44% | 3.56 pp | -19 | 21 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 267 | 118 | 149 | 44.19% | 43.33% | 44.19% | 5.81 pp | -31 | 22 | -1.41 |
| Consolidated Market Hours | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 267 | 118 | 149 | 44.19% | 42.50% | 44.19% | 5.81 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 267 | 114 | 153 | 42.70% | 41.25% | 42.70% | 7.30 pp | -39 | 22 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 269 | 123 | 146 | 45.72% | 44.58% | 45.72% | 4.28 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 269 | 123 | 146 | 45.72% | 45.00% | 45.72% | 4.28 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| BTC Hourly | transformer | Transformer | 244 | 110 | 134 | 45.08% | 45.83% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 267 | 109 | 158 | 40.82% | 42.08% | 40.82% | 9.18 pp | -49 | 22 | -2.23 |
| Consolidated Market Hours | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| BTC Market Hours | lstm | LSTM | 267 | 109 | 158 | 40.82% | 42.50% | 40.82% | 9.18 pp | -49 | 21 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 244 | 102 | 142 | 41.80% | 42.08% | 41.80% | 8.20 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 244 | 100 | 144 | 40.98% | 41.67% | 40.98% | 9.02 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 269 | 108 | 161 | 40.15% | 38.33% | 40.15% | 9.85 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 269 | 102 | 167 | 37.92% | 37.50% | 37.92% | 12.08 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 279 | 103 | 176 | 36.92% | 37.50% | 36.92% | 13.08 pp | -73 | 13 | -5.62 |
| BTC Hourly | lstm | LSTM | 244 | 90 | 154 | 36.89% | 37.08% | 36.89% | 13.11 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 269 | 96 | 173 | 35.69% | 35.83% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 269 | 123 | 146 | 45.72% | 44.58% | 45.72% | 4.28 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 269 | 123 | 146 | 45.72% | 45.00% | 45.72% | 4.28 pp | -23 | 12 | -1.92 |
| BTC Daily | transformer | Transformer | 269 | 108 | 161 | 40.15% | 38.33% | 40.15% | 9.85 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 269 | 102 | 167 | 37.92% | 37.50% | 37.92% | 12.08 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 279 | 103 | 176 | 36.92% | 37.50% | 36.92% | 13.08 pp | -73 | 13 | -5.62 |
| BTC Daily | lstm | LSTM | 269 | 96 | 173 | 35.69% | 35.83% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 267 | 141 | 126 | 52.81% | 52.50% | 52.81% | 2.81 pp | 15 | 21 | 0.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 267 | 125 | 142 | 46.82% | 47.08% | 46.82% | 3.18 pp | -17 | 21 | -0.81 |
| BTC Market Hours | transformer | Transformer | 267 | 125 | 142 | 46.82% | 46.67% | 46.82% | 3.18 pp | -17 | 21 | -0.81 |
| BTC Market Hours | xgb | XGBoost | 267 | 124 | 143 | 46.44% | 45.42% | 46.44% | 3.56 pp | -19 | 21 | -0.90 |
| BTC Market Hours | rf | RandomForest | 267 | 118 | 149 | 44.19% | 42.50% | 44.19% | 5.81 pp | -31 | 21 | -1.48 |
| BTC Market Hours | lstm | LSTM | 267 | 109 | 158 | 40.82% | 42.50% | 40.82% | 9.18 pp | -49 | 21 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 267 | 132 | 135 | 49.44% | 48.75% | 49.44% | 0.56 pp | -3 | 22 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 267 | 130 | 137 | 48.69% | 48.33% | 48.69% | 1.31 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | nn | NN | 267 | 129 | 138 | 48.31% | 48.75% | 48.31% | 1.69 pp | -9 | 22 | -0.41 |
| BTC Market Hours Daily | xgb | XGBoost | 267 | 118 | 149 | 44.19% | 43.33% | 44.19% | 5.81 pp | -31 | 22 | -1.41 |
| BTC Market Hours Daily | rf | RandomForest | 267 | 114 | 153 | 42.70% | 41.25% | 42.70% | 7.30 pp | -39 | 22 | -1.77 |
| BTC Market Hours Daily | lstm | LSTM | 267 | 109 | 158 | 40.82% | 42.08% | 40.82% | 9.18 pp | -49 | 22 | -2.23 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Hourly | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 233 | 113 | 120 | 48.50% | 48.50% | 48.50% | 1.50 pp | -7 | 15 | -0.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 233 | 92 | 141 | 39.48% | 39.48% | 39.48% | 10.52 pp | -49 | 15 | -3.27 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
