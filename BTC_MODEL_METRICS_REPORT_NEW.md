# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T00:04:22.243758+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 283 | 223 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 319 | 259 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 23:00:00+00:00 | 467 | 247 | 220 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 23:00:00+00:00 | 467 | 247 | 220 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 215 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 247 | 127 | 120 | 51.42% | 52.08% | 51.42% | 1.42 pp | 7 | 19 | 0.37 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 223 | 111 | 112 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 247 | 121 | 126 | 48.99% | 48.75% | 48.99% | 1.01 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 247 | 120 | 127 | 48.58% | 48.75% | 48.58% | 1.42 pp | -7 | 20 | -0.35 |
| BTC Market Hours Daily | nn | NN | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 20 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 247 | 117 | 130 | 47.37% | 47.92% | 47.37% | 2.63 pp | -13 | 19 | -0.68 |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 247 | 116 | 131 | 46.96% | 46.67% | 46.96% | 3.04 pp | -15 | 19 | -0.79 |
| BTC Market Hours | transformer | Transformer | 247 | 115 | 132 | 46.56% | 46.67% | 46.56% | 3.44 pp | -17 | 19 | -0.89 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Market Hours | rf | RandomForest | 247 | 112 | 135 | 45.34% | 45.42% | 45.34% | 4.66 pp | -23 | 19 | -1.21 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 247 | 110 | 137 | 44.53% | 44.17% | 44.53% | 5.47 pp | -27 | 20 | -1.35 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 249 | 116 | 133 | 46.59% | 47.08% | 46.59% | 3.41 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 247 | 108 | 139 | 43.72% | 42.92% | 43.72% | 6.28 pp | -31 | 20 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 249 | 114 | 135 | 45.78% | 45.42% | 45.78% | 4.22 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 247 | 105 | 142 | 42.51% | 43.33% | 42.51% | 7.49 pp | -37 | 19 | -1.95 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 20 | -2.35 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Hourly | transformer | Transformer | 223 | 98 | 125 | 43.95% | 43.95% | 43.95% | 6.05 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |
| BTC Hourly | nn | NN | 223 | 93 | 130 | 41.70% | 41.70% | 41.70% | 8.30 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 223 | 92 | 131 | 41.26% | 41.26% | 41.26% | 8.74 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 249 | 100 | 149 | 40.16% | 39.58% | 40.16% | 9.84 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 249 | 95 | 154 | 38.15% | 37.08% | 38.15% | 11.85 pp | -59 | 11 | -5.36 |
| BTC Hourly | lstm | LSTM | 223 | 83 | 140 | 37.22% | 37.22% | 37.22% | 12.78 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 259 | 92 | 167 | 35.52% | 35.00% | 35.52% | 14.48 pp | -75 | 12 | -6.25 |
| BTC Hourly | xgb | XGBoost | 223 | 78 | 145 | 34.98% | 34.98% | 34.98% | 15.02 pp | -67 | 10 | -6.70 |
| BTC Daily | lstm | LSTM | 249 | 85 | 164 | 34.14% | 34.58% | 34.14% | 15.86 pp | -79 | 11 | -7.18 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 223 | 111 | 112 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Hourly | transformer | Transformer | 223 | 98 | 125 | 43.95% | 43.95% | 43.95% | 6.05 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 223 | 93 | 130 | 41.70% | 41.70% | 41.70% | 8.30 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 223 | 92 | 131 | 41.26% | 41.26% | 41.26% | 8.74 pp | -39 | 10 | -3.90 |
| BTC Hourly | lstm | LSTM | 223 | 83 | 140 | 37.22% | 37.22% | 37.22% | 12.78 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 223 | 78 | 145 | 34.98% | 34.98% | 34.98% | 15.02 pp | -67 | 10 | -6.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 249 | 116 | 133 | 46.59% | 47.08% | 46.59% | 3.41 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 249 | 114 | 135 | 45.78% | 45.42% | 45.78% | 4.22 pp | -21 | 11 | -1.91 |
| BTC Daily | transformer | Transformer | 249 | 100 | 149 | 40.16% | 39.58% | 40.16% | 9.84 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 249 | 95 | 154 | 38.15% | 37.08% | 38.15% | 11.85 pp | -59 | 11 | -5.36 |
| BTC Daily | xgb | XGBoost | 259 | 92 | 167 | 35.52% | 35.00% | 35.52% | 14.48 pp | -75 | 12 | -6.25 |
| BTC Daily | lstm | LSTM | 249 | 85 | 164 | 34.14% | 34.58% | 34.14% | 15.86 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 247 | 127 | 120 | 51.42% | 52.08% | 51.42% | 1.42 pp | 7 | 19 | 0.37 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 247 | 117 | 130 | 47.37% | 47.92% | 47.37% | 2.63 pp | -13 | 19 | -0.68 |
| BTC Market Hours | xgb | XGBoost | 247 | 116 | 131 | 46.96% | 46.67% | 46.96% | 3.04 pp | -15 | 19 | -0.79 |
| BTC Market Hours | transformer | Transformer | 247 | 115 | 132 | 46.56% | 46.67% | 46.56% | 3.44 pp | -17 | 19 | -0.89 |
| BTC Market Hours | rf | RandomForest | 247 | 112 | 135 | 45.34% | 45.42% | 45.34% | 4.66 pp | -23 | 19 | -1.21 |
| BTC Market Hours | lstm | LSTM | 247 | 105 | 142 | 42.51% | 43.33% | 42.51% | 7.49 pp | -37 | 19 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 247 | 121 | 126 | 48.99% | 48.75% | 48.99% | 1.01 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 247 | 120 | 127 | 48.58% | 48.75% | 48.58% | 1.42 pp | -7 | 20 | -0.35 |
| BTC Market Hours Daily | nn | NN | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 20 | -0.55 |
| BTC Market Hours Daily | xgb | XGBoost | 247 | 110 | 137 | 44.53% | 44.17% | 44.53% | 5.47 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | rf | RandomForest | 247 | 108 | 139 | 43.72% | 42.92% | 43.72% | 6.28 pp | -31 | 20 | -1.55 |
| BTC Market Hours Daily | lstm | LSTM | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 20 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 14 | -0.93 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 215 | 98 | 117 | 45.58% | 45.58% | 45.58% | 4.42 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 215 | 93 | 122 | 43.26% | 43.26% | 43.26% | 6.74 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
