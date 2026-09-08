# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T00:31:18.855382+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 438 | 231 | 207 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 23:00:00+00:00 | 438 | 231 | 207 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 201 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T22:00:00+00:00 | 202 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 231 | 119 | 112 | 51.52% | 51.52% | 51.52% | 1.52 pp | 7 | 18 | 0.39 |
| BTC Market Hours Daily | transformer | Transformer | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 231 | 114 | 117 | 49.35% | 49.35% | 49.35% | 0.65 pp | -3 | 19 | -0.16 |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 231 | 111 | 120 | 48.05% | 48.05% | 48.05% | 1.95 pp | -9 | 19 | -0.47 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 231 | 110 | 121 | 47.62% | 47.62% | 47.62% | 2.38 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 231 | 109 | 122 | 47.19% | 47.19% | 47.19% | 2.81 pp | -13 | 18 | -0.72 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 30 | 35 | 46.15% | 46.15% | 46.15% | 3.85 pp | -5 | 5 | -1.00 |
| BTC Market Hours | rf | RandomForest | 231 | 106 | 125 | 45.89% | 45.89% | 45.89% | 4.11 pp | -19 | 18 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| BTC Market Hours | xgb | XGBoost | 231 | 105 | 126 | 45.45% | 45.45% | 45.45% | 4.55 pp | -21 | 18 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 19 | -1.32 |
| BTC Daily | mlp_sklearn | MLPClassifier | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 11 | -1.36 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 19 | -1.74 |
| Consolidated Hourly | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |
| BTC Market Hours | lstm | LSTM | 231 | 97 | 134 | 41.99% | 41.99% | 41.99% | 8.01 pp | -37 | 18 | -2.06 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 233 | 104 | 129 | 44.64% | 44.64% | 44.64% | 5.36 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 231 | 93 | 138 | 40.26% | 40.26% | 40.26% | 9.74 pp | -45 | 19 | -2.37 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
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
| BTC Market Hours Daily | transformer | Transformer | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 231 | 114 | 117 | 49.35% | 49.35% | 49.35% | 0.65 pp | -3 | 19 | -0.16 |
| BTC Market Hours Daily | nn | NN | 231 | 111 | 120 | 48.05% | 48.05% | 48.05% | 1.95 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | rf | RandomForest | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 231 | 99 | 132 | 42.86% | 42.86% | 42.86% | 7.14 pp | -33 | 19 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 231 | 93 | 138 | 40.26% | 40.26% | 40.26% | 9.74 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 99 | 102 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 13 | -1.92 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 30 | 35 | 46.15% | 46.15% | 46.15% | 3.85 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 5 | -2.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
