# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T22:26:19.327461+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 282 | 222 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 318 | 258 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 21:00:00+00:00 | 464 | 246 | 218 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 21:00:00+00:00 | 464 | 246 | 218 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 213 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 16:00:00+00:00 | 213 | 71 | 142 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 246 | 126 | 120 | 51.22% | 51.67% | 51.22% | 1.22 pp | 6 | 19 | 0.32 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 222 | 110 | 112 | 49.55% | 49.55% | 49.55% | 0.45 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| BTC Market Hours Daily | nn | NN | 246 | 117 | 129 | 47.56% | 47.92% | 47.56% | 2.44 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 246 | 116 | 130 | 47.15% | 47.50% | 47.15% | 2.85 pp | -14 | 19 | -0.74 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Market Hours | xgb | XGBoost | 246 | 115 | 131 | 46.75% | 46.25% | 46.75% | 3.25 pp | -16 | 19 | -0.84 |
| BTC Market Hours | transformer | Transformer | 246 | 114 | 132 | 46.34% | 46.67% | 46.34% | 3.66 pp | -18 | 19 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 246 | 111 | 135 | 45.12% | 45.00% | 45.12% | 4.88 pp | -24 | 19 | -1.26 |
| BTC Market Hours Daily | xgb | XGBoost | 246 | 109 | 137 | 44.31% | 43.75% | 44.31% | 5.69 pp | -28 | 20 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 248 | 116 | 132 | 46.77% | 47.08% | 46.77% | 3.23 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 246 | 107 | 139 | 43.50% | 42.50% | 43.50% | 6.50 pp | -32 | 20 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 11 | -2.00 |
| BTC Market Hours | lstm | LSTM | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 19 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 246 | 100 | 146 | 40.65% | 41.25% | 40.65% | 9.35 pp | -46 | 20 | -2.30 |
| Consolidated Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| BTC Hourly | transformer | Transformer | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 248 | 99 | 149 | 39.92% | 39.17% | 39.92% | 10.08 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 248 | 95 | 153 | 38.31% | 37.08% | 38.31% | 11.69 pp | -58 | 11 | -5.27 |
| BTC Hourly | lstm | LSTM | 222 | 82 | 140 | 36.94% | 36.94% | 36.94% | 13.06 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 258 | 91 | 167 | 35.27% | 34.58% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 10 | -6.80 |
| BTC Daily | lstm | LSTM | 248 | 85 | 163 | 34.27% | 34.58% | 34.27% | 15.73 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 222 | 110 | 112 | 49.55% | 49.55% | 49.55% | 0.45 pp | -2 | 10 | -0.20 |
| BTC Hourly | transformer | Transformer | 222 | 98 | 124 | 44.14% | 44.14% | 44.14% | 5.86 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 222 | 93 | 129 | 41.89% | 41.89% | 41.89% | 8.11 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 222 | 92 | 130 | 41.44% | 41.44% | 41.44% | 8.56 pp | -38 | 10 | -3.80 |
| BTC Hourly | lstm | LSTM | 222 | 82 | 140 | 36.94% | 36.94% | 36.94% | 13.06 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 222 | 77 | 145 | 34.68% | 34.68% | 34.68% | 15.32 pp | -68 | 10 | -6.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 248 | 116 | 132 | 46.77% | 47.08% | 46.77% | 3.23 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 248 | 113 | 135 | 45.56% | 45.42% | 45.56% | 4.44 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 248 | 99 | 149 | 39.92% | 39.17% | 39.92% | 10.08 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 248 | 95 | 153 | 38.31% | 37.08% | 38.31% | 11.69 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 258 | 91 | 167 | 35.27% | 34.58% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 248 | 85 | 163 | 34.27% | 34.58% | 34.27% | 15.73 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 246 | 126 | 120 | 51.22% | 51.67% | 51.22% | 1.22 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 246 | 116 | 130 | 47.15% | 47.50% | 47.15% | 2.85 pp | -14 | 19 | -0.74 |
| BTC Market Hours | xgb | XGBoost | 246 | 115 | 131 | 46.75% | 46.25% | 46.75% | 3.25 pp | -16 | 19 | -0.84 |
| BTC Market Hours | transformer | Transformer | 246 | 114 | 132 | 46.34% | 46.67% | 46.34% | 3.66 pp | -18 | 19 | -0.95 |
| BTC Market Hours | rf | RandomForest | 246 | 111 | 135 | 45.12% | 45.00% | 45.12% | 4.88 pp | -24 | 19 | -1.26 |
| BTC Market Hours | lstm | LSTM | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 19 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 246 | 117 | 129 | 47.56% | 47.92% | 47.56% | 2.44 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 246 | 109 | 137 | 44.31% | 43.75% | 44.31% | 5.69 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 246 | 107 | 139 | 43.50% | 42.50% | 43.50% | 6.50 pp | -32 | 20 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 246 | 100 | 146 | 40.65% | 41.25% | 40.65% | 9.35 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 99 | 114 | 46.48% | 46.48% | 46.48% | 3.52 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 94 | 119 | 44.13% | 44.13% | 44.13% | 5.87 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
