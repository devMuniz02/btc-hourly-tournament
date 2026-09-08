# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T19:47:05.705280+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 280 | 220 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 316 | 256 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 18:00:00+00:00 | 459 | 244 | 215 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 18:00:00+00:00 | 459 | 244 | 215 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 214 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 244 | 125 | 119 | 51.23% | 51.67% | 51.23% | 1.23 pp | 6 | 19 | 0.32 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 244 | 119 | 125 | 48.77% | 48.75% | 48.77% | 1.23 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 244 | 119 | 125 | 48.77% | 48.75% | 48.77% | 1.23 pp | -6 | 20 | -0.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 220 | 108 | 112 | 49.09% | 49.09% | 49.09% | 0.91 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | nn | NN | 244 | 116 | 128 | 47.54% | 47.92% | 47.54% | 2.46 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 244 | 116 | 128 | 47.54% | 47.50% | 47.54% | 2.46 pp | -12 | 19 | -0.63 |
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Market Hours | xgb | XGBoost | 244 | 114 | 130 | 46.72% | 46.25% | 46.72% | 3.28 pp | -16 | 19 | -0.84 |
| BTC Market Hours | transformer | Transformer | 244 | 113 | 131 | 46.31% | 46.67% | 46.31% | 3.69 pp | -18 | 19 | -0.95 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 244 | 110 | 134 | 45.08% | 45.00% | 45.08% | 4.92 pp | -24 | 19 | -1.26 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 244 | 108 | 136 | 44.26% | 43.75% | 44.26% | 5.74 pp | -28 | 20 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 246 | 115 | 131 | 46.75% | 47.08% | 46.75% | 3.25 pp | -16 | 11 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 244 | 107 | 137 | 43.85% | 43.33% | 43.85% | 6.15 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 11 | -2.00 |
| BTC Market Hours | lstm | LSTM | 244 | 103 | 141 | 42.21% | 42.50% | 42.21% | 7.79 pp | -38 | 19 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 244 | 99 | 145 | 40.57% | 41.25% | 40.57% | 9.43 pp | -46 | 20 | -2.30 |
| BTC Hourly | transformer | Transformer | 220 | 98 | 122 | 44.55% | 44.55% | 44.55% | 5.45 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 220 | 93 | 127 | 42.27% | 42.27% | 42.27% | 7.73 pp | -34 | 10 | -3.40 |
| BTC Hourly | rf | RandomForest | 220 | 92 | 128 | 41.82% | 41.82% | 41.82% | 8.18 pp | -36 | 10 | -3.60 |
| BTC Daily | transformer | Transformer | 246 | 99 | 147 | 40.24% | 39.58% | 40.24% | 9.76 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 246 | 95 | 151 | 38.62% | 37.92% | 38.62% | 11.38 pp | -56 | 11 | -5.09 |
| BTC Hourly | lstm | LSTM | 220 | 82 | 138 | 37.27% | 37.27% | 37.27% | 12.73 pp | -56 | 10 | -5.60 |
| BTC Daily | xgb | XGBoost | 256 | 91 | 165 | 35.55% | 35.42% | 35.55% | 14.45 pp | -74 | 12 | -6.17 |
| BTC Hourly | xgb | XGBoost | 220 | 77 | 143 | 35.00% | 35.00% | 35.00% | 15.00 pp | -66 | 10 | -6.60 |
| BTC Daily | lstm | LSTM | 246 | 84 | 162 | 34.15% | 34.17% | 34.15% | 15.85 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 220 | 108 | 112 | 49.09% | 49.09% | 49.09% | 0.91 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 220 | 98 | 122 | 44.55% | 44.55% | 44.55% | 5.45 pp | -24 | 10 | -2.40 |
| BTC Hourly | nn | NN | 220 | 93 | 127 | 42.27% | 42.27% | 42.27% | 7.73 pp | -34 | 10 | -3.40 |
| BTC Hourly | rf | RandomForest | 220 | 92 | 128 | 41.82% | 41.82% | 41.82% | 8.18 pp | -36 | 10 | -3.60 |
| BTC Hourly | lstm | LSTM | 220 | 82 | 138 | 37.27% | 37.27% | 37.27% | 12.73 pp | -56 | 10 | -5.60 |
| BTC Hourly | xgb | XGBoost | 220 | 77 | 143 | 35.00% | 35.00% | 35.00% | 15.00 pp | -66 | 10 | -6.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 246 | 115 | 131 | 46.75% | 47.08% | 46.75% | 3.25 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 11 | -2.00 |
| BTC Daily | transformer | Transformer | 246 | 99 | 147 | 40.24% | 39.58% | 40.24% | 9.76 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 246 | 95 | 151 | 38.62% | 37.92% | 38.62% | 11.38 pp | -56 | 11 | -5.09 |
| BTC Daily | xgb | XGBoost | 256 | 91 | 165 | 35.55% | 35.42% | 35.55% | 14.45 pp | -74 | 12 | -6.17 |
| BTC Daily | lstm | LSTM | 246 | 84 | 162 | 34.15% | 34.17% | 34.15% | 15.85 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 244 | 125 | 119 | 51.23% | 51.67% | 51.23% | 1.23 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 244 | 116 | 128 | 47.54% | 47.50% | 47.54% | 2.46 pp | -12 | 19 | -0.63 |
| BTC Market Hours | xgb | XGBoost | 244 | 114 | 130 | 46.72% | 46.25% | 46.72% | 3.28 pp | -16 | 19 | -0.84 |
| BTC Market Hours | transformer | Transformer | 244 | 113 | 131 | 46.31% | 46.67% | 46.31% | 3.69 pp | -18 | 19 | -0.95 |
| BTC Market Hours | rf | RandomForest | 244 | 110 | 134 | 45.08% | 45.00% | 45.08% | 4.92 pp | -24 | 19 | -1.26 |
| BTC Market Hours | lstm | LSTM | 244 | 103 | 141 | 42.21% | 42.50% | 42.21% | 7.79 pp | -38 | 19 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 244 | 119 | 125 | 48.77% | 48.75% | 48.77% | 1.23 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 244 | 119 | 125 | 48.77% | 48.75% | 48.77% | 1.23 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 244 | 116 | 128 | 47.54% | 47.92% | 47.54% | 2.46 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 244 | 108 | 136 | 44.26% | 43.75% | 44.26% | 5.74 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 244 | 107 | 137 | 43.85% | 43.33% | 43.85% | 6.15 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 244 | 99 | 145 | 40.57% | 41.25% | 40.57% | 9.43 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
