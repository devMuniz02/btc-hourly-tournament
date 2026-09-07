# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T23:18:29.913782+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 266 | 206 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 302 | 242 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 436 | 230 | 206 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 436 | 230 | 206 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 230 | 119 | 111 | 51.74% | 51.74% | 51.74% | 1.74 pp | 8 | 18 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 206 | 103 | 103 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 230 | 114 | 116 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 230 | 113 | 117 | 49.13% | 49.13% | 49.13% | 0.87 pp | -4 | 19 | -0.21 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 19 | -0.53 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 230 | 109 | 121 | 47.39% | 47.39% | 47.39% | 2.61 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 18 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Market Hours | xgb | XGBoost | 230 | 104 | 126 | 45.22% | 45.22% | 45.22% | 4.78 pp | -22 | 18 | -1.22 |
| BTC Market Hours Daily | rf | RandomForest | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 19 | -1.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 232 | 108 | 124 | 46.55% | 46.55% | 46.55% | 3.45 pp | -16 | 10 | -1.60 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | xgb | XGBoost | 230 | 98 | 132 | 42.61% | 42.61% | 42.61% | 7.39 pp | -34 | 19 | -1.79 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 18 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 230 | 93 | 137 | 40.43% | 40.43% | 40.43% | 9.57 pp | -44 | 19 | -2.32 |
| BTC Daily | nn | NN | 232 | 104 | 128 | 44.83% | 44.83% | 44.83% | 5.17 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 206 | 87 | 119 | 42.23% | 42.23% | 42.23% | 7.77 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 206 | 85 | 121 | 41.26% | 41.26% | 41.26% | 8.74 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 232 | 89 | 143 | 38.36% | 38.36% | 38.36% | 11.64 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 206 | 77 | 129 | 37.38% | 37.38% | 37.38% | 12.62 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 242 | 86 | 156 | 35.54% | 35.42% | 35.54% | 14.46 pp | -70 | 11 | -6.36 |
| BTC Hourly | xgb | XGBoost | 206 | 71 | 135 | 34.47% | 34.47% | 34.47% | 15.53 pp | -64 | 9 | -7.11 |
| BTC Daily | lstm | LSTM | 232 | 77 | 155 | 33.19% | 33.19% | 33.19% | 16.81 pp | -78 | 10 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 206 | 103 | 103 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 206 | 87 | 119 | 42.23% | 42.23% | 42.23% | 7.77 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 206 | 85 | 121 | 41.26% | 41.26% | 41.26% | 8.74 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 206 | 77 | 129 | 37.38% | 37.38% | 37.38% | 12.62 pp | -52 | 9 | -5.78 |
| BTC Hourly | xgb | XGBoost | 206 | 71 | 135 | 34.47% | 34.47% | 34.47% | 15.53 pp | -64 | 9 | -7.11 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 232 | 108 | 124 | 46.55% | 46.55% | 46.55% | 3.45 pp | -16 | 10 | -1.60 |
| BTC Daily | nn | NN | 232 | 104 | 128 | 44.83% | 44.83% | 44.83% | 5.17 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 232 | 89 | 143 | 38.36% | 38.36% | 38.36% | 11.64 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 242 | 86 | 156 | 35.54% | 35.42% | 35.54% | 14.46 pp | -70 | 11 | -6.36 |
| BTC Daily | lstm | LSTM | 232 | 77 | 155 | 33.19% | 33.19% | 33.19% | 16.81 pp | -78 | 10 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 230 | 119 | 111 | 51.74% | 51.74% | 51.74% | 1.74 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 230 | 110 | 120 | 47.83% | 47.83% | 47.83% | 2.17 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 230 | 109 | 121 | 47.39% | 47.39% | 47.39% | 2.61 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 230 | 104 | 126 | 45.22% | 45.22% | 45.22% | 4.78 pp | -22 | 18 | -1.22 |
| BTC Market Hours | lstm | LSTM | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 18 | -2.11 |

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
