# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T07:25:52.056566+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 303 | 243 | 60 | 0 |
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
| BTC Hourly | mlp_sklearn | MLPClassifier | 243 | 118 | 125 | 48.56% | 48.75% | 48.56% | 1.44 pp | -7 | 11 | -0.64 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 269 | 124 | 145 | 46.10% | 45.00% | 46.10% | 3.90 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 269 | 124 | 145 | 46.10% | 45.42% | 46.10% | 3.90 pp | -21 | 12 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 267 | 114 | 153 | 42.70% | 41.25% | 42.70% | 7.30 pp | -39 | 22 | -1.77 |
| Consolidated Hourly | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 233 | 102 | 131 | 43.78% | 43.78% | 43.78% | 6.22 pp | -29 | 15 | -1.93 |
| BTC Hourly | transformer | Transformer | 243 | 110 | 133 | 45.27% | 45.83% | 45.27% | 4.73 pp | -23 | 11 | -2.09 |
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
| BTC Hourly | nn | NN | 243 | 102 | 141 | 41.98% | 42.08% | 41.98% | 8.02 pp | -39 | 11 | -3.55 |
| BTC Hourly | rf | RandomForest | 243 | 100 | 143 | 41.15% | 41.67% | 41.15% | 8.85 pp | -43 | 11 | -3.91 |
| BTC Daily | transformer | Transformer | 269 | 109 | 160 | 40.52% | 38.75% | 40.52% | 9.48 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 269 | 103 | 166 | 38.29% | 37.92% | 38.29% | 11.71 pp | -63 | 12 | -5.25 |
| BTC Daily | xgb | XGBoost | 279 | 104 | 175 | 37.28% | 37.92% | 37.28% | 12.72 pp | -71 | 13 | -5.46 |
| BTC Hourly | lstm | LSTM | 243 | 90 | 153 | 37.04% | 37.08% | 37.04% | 12.96 pp | -63 | 11 | -5.73 |
| BTC Daily | lstm | LSTM | 269 | 96 | 173 | 35.69% | 36.25% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 243 | 85 | 158 | 34.98% | 35.42% | 34.98% | 15.02 pp | -73 | 11 | -6.64 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 243 | 118 | 125 | 48.56% | 48.75% | 48.56% | 1.44 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 243 | 110 | 133 | 45.27% | 45.83% | 45.27% | 4.73 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 243 | 102 | 141 | 41.98% | 42.08% | 41.98% | 8.02 pp | -39 | 11 | -3.55 |
| BTC Hourly | rf | RandomForest | 243 | 100 | 143 | 41.15% | 41.67% | 41.15% | 8.85 pp | -43 | 11 | -3.91 |
| BTC Hourly | lstm | LSTM | 243 | 90 | 153 | 37.04% | 37.08% | 37.04% | 12.96 pp | -63 | 11 | -5.73 |
| BTC Hourly | xgb | XGBoost | 243 | 85 | 158 | 34.98% | 35.42% | 34.98% | 15.02 pp | -73 | 11 | -6.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 269 | 124 | 145 | 46.10% | 45.00% | 46.10% | 3.90 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 269 | 124 | 145 | 46.10% | 45.42% | 46.10% | 3.90 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 269 | 109 | 160 | 40.52% | 38.75% | 40.52% | 9.48 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 269 | 103 | 166 | 38.29% | 37.92% | 38.29% | 11.71 pp | -63 | 12 | -5.25 |
| BTC Daily | xgb | XGBoost | 279 | 104 | 175 | 37.28% | 37.92% | 37.28% | 12.72 pp | -71 | 13 | -5.46 |
| BTC Daily | lstm | LSTM | 269 | 96 | 173 | 35.69% | 36.25% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |

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
