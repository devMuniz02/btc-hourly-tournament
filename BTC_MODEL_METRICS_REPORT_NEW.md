# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T23:37:31.675899+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 267 | 207 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 303 | 243 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 437 | 231 | 206 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 436 | 230 | 206 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 231 | 119 | 112 | 51.52% | 51.52% | 51.52% | 1.52 pp | 7 | 18 | 0.39 |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 230 | 114 | 116 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 230 | 113 | 117 | 49.13% | 49.13% | 49.13% | 0.87 pp | -4 | 19 | -0.21 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 19 | -0.53 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 231 | 109 | 122 | 47.19% | 47.19% | 47.19% | 2.81 pp | -13 | 18 | -0.72 |
| BTC Market Hours | rf | RandomForest | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 18 | -1.06 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| BTC Market Hours | xgb | XGBoost | 231 | 105 | 126 | 45.45% | 45.45% | 45.45% | 4.55 pp | -21 | 18 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 19 | -1.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | xgb | XGBoost | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 19 | -1.79 |
| BTC Market Hours | lstm | LSTM | 231 | 97 | 134 | 41.99% | 41.99% | 41.99% | 8.01 pp | -37 | 18 | -2.06 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| BTC Daily | nn | NN | 233 | 104 | 129 | 44.64% | 44.64% | 44.64% | 5.36 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 230 | 93 | 137 | 40.43% | 40.43% | 40.43% | 9.57 pp | -44 | 19 | -2.32 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 9 | -2.56 |
| BTC Hourly | nn | NN | 207 | 87 | 120 | 42.03% | 42.03% | 42.03% | 7.97 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 207 | 85 | 122 | 41.06% | 41.06% | 41.06% | 8.94 pp | -37 | 9 | -4.11 |
| BTC Daily | transformer | Transformer | 233 | 93 | 140 | 39.91% | 39.91% | 39.91% | 10.09 pp | -47 | 11 | -4.27 |
| BTC Daily | rf | RandomForest | 233 | 90 | 143 | 38.63% | 38.63% | 38.63% | 11.37 pp | -53 | 11 | -4.82 |
| BTC Hourly | lstm | LSTM | 207 | 77 | 130 | 37.20% | 37.20% | 37.20% | 12.80 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 243 | 86 | 157 | 35.39% | 35.42% | 35.39% | 14.61 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 233 | 77 | 156 | 33.05% | 33.05% | 33.05% | 16.95 pp | -79 | 11 | -7.18 |
| BTC Hourly | xgb | XGBoost | 207 | 71 | 136 | 34.30% | 34.30% | 34.30% | 15.70 pp | -65 | 9 | -7.22 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 9 | -2.56 |
| BTC Hourly | nn | NN | 207 | 87 | 120 | 42.03% | 42.03% | 42.03% | 7.97 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 207 | 85 | 122 | 41.06% | 41.06% | 41.06% | 8.94 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 207 | 77 | 130 | 37.20% | 37.20% | 37.20% | 12.80 pp | -53 | 9 | -5.89 |
| BTC Hourly | xgb | XGBoost | 207 | 71 | 136 | 34.30% | 34.30% | 34.30% | 15.70 pp | -65 | 9 | -7.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 233 | 104 | 129 | 44.64% | 44.64% | 44.64% | 5.36 pp | -25 | 11 | -2.27 |
| BTC Daily | transformer | Transformer | 233 | 93 | 140 | 39.91% | 39.91% | 39.91% | 10.09 pp | -47 | 11 | -4.27 |
| BTC Daily | rf | RandomForest | 233 | 90 | 143 | 38.63% | 38.63% | 38.63% | 11.37 pp | -53 | 11 | -4.82 |
| BTC Daily | xgb | XGBoost | 243 | 86 | 157 | 35.39% | 35.42% | 35.39% | 14.61 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 233 | 77 | 156 | 33.05% | 33.05% | 33.05% | 16.95 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 231 | 119 | 112 | 51.52% | 51.52% | 51.52% | 1.52 pp | 7 | 18 | 0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 231 | 109 | 122 | 47.19% | 47.19% | 47.19% | 2.81 pp | -13 | 18 | -0.72 |
| BTC Market Hours | rf | RandomForest | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 18 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 231 | 105 | 126 | 45.45% | 45.45% | 45.45% | 4.55 pp | -21 | 18 | -1.17 |
| BTC Market Hours | lstm | LSTM | 231 | 97 | 134 | 41.99% | 41.99% | 41.99% | 8.01 pp | -37 | 18 | -2.06 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 230 | 114 | 116 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 230 | 113 | 117 | 49.13% | 49.13% | 49.13% | 0.87 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | nn | NN | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 19 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 19 | -1.26 |
| BTC Market Hours Daily | xgb | XGBoost | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 19 | -1.79 |
| BTC Market Hours Daily | lstm | LSTM | 230 | 93 | 137 | 40.43% | 40.43% | 40.43% | 9.57 pp | -44 | 19 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
