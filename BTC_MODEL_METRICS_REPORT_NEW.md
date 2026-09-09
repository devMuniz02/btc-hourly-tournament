# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T22:35:15.216184+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 333 | 273 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 492 | 261 | 231 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 21:00:00+00:00 | 492 | 261 | 231 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 227 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 00:00:00+00:00 | 227 | 79 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 261 | 137 | 124 | 52.49% | 52.50% | 52.49% | 2.49 pp | 13 | 21 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 261 | 130 | 131 | 49.81% | 49.58% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 261 | 128 | 133 | 49.04% | 48.75% | 49.04% | 0.96 pp | -5 | 21 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 261 | 125 | 136 | 47.89% | 48.33% | 47.89% | 2.11 pp | -11 | 21 | -0.52 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 238 | 116 | 122 | 48.74% | 48.74% | 48.74% | 1.26 pp | -6 | 10 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 261 | 123 | 138 | 47.13% | 47.92% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 261 | 123 | 138 | 47.13% | 47.50% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | xgb | XGBoost | 261 | 120 | 141 | 45.98% | 44.17% | 45.98% | 4.02 pp | -21 | 21 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 261 | 116 | 145 | 44.44% | 43.33% | 44.44% | 5.56 pp | -29 | 21 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 261 | 115 | 146 | 44.06% | 42.92% | 44.06% | 5.94 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 261 | 113 | 148 | 43.30% | 42.50% | 43.30% | 6.70 pp | -35 | 21 | -1.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 263 | 121 | 142 | 46.01% | 45.42% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 263 | 121 | 142 | 46.01% | 45.42% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 261 | 108 | 153 | 41.38% | 42.08% | 41.38% | 8.62 pp | -45 | 21 | -2.14 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| BTC Market Hours | lstm | LSTM | 261 | 106 | 155 | 40.61% | 42.08% | 40.61% | 9.39 pp | -49 | 21 | -2.33 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| BTC Hourly | transformer | Transformer | 238 | 106 | 132 | 44.54% | 44.54% | 44.54% | 5.46 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 238 | 100 | 138 | 42.02% | 42.02% | 42.02% | 7.98 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 263 | 107 | 156 | 40.68% | 40.00% | 40.68% | 9.32 pp | -49 | 12 | -4.08 |
| BTC Hourly | rf | RandomForest | 238 | 98 | 140 | 41.18% | 41.18% | 41.18% | 8.82 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 263 | 98 | 165 | 37.26% | 37.08% | 37.26% | 12.74 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 273 | 99 | 174 | 36.26% | 36.67% | 36.26% | 13.74 pp | -75 | 13 | -5.77 |
| BTC Hourly | lstm | LSTM | 238 | 88 | 150 | 36.97% | 36.97% | 36.97% | 13.03 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 263 | 93 | 170 | 35.36% | 35.83% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 10 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 238 | 116 | 122 | 48.74% | 48.74% | 48.74% | 1.26 pp | -6 | 10 | -0.60 |
| BTC Hourly | transformer | Transformer | 238 | 106 | 132 | 44.54% | 44.54% | 44.54% | 5.46 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 238 | 100 | 138 | 42.02% | 42.02% | 42.02% | 7.98 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 238 | 98 | 140 | 41.18% | 41.18% | 41.18% | 8.82 pp | -42 | 10 | -4.20 |
| BTC Hourly | lstm | LSTM | 238 | 88 | 150 | 36.97% | 36.97% | 36.97% | 13.03 pp | -62 | 10 | -6.20 |
| BTC Hourly | xgb | XGBoost | 238 | 81 | 157 | 34.03% | 34.03% | 34.03% | 15.97 pp | -76 | 10 | -7.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 263 | 121 | 142 | 46.01% | 45.42% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| BTC Daily | nn | NN | 263 | 121 | 142 | 46.01% | 45.42% | 46.01% | 3.99 pp | -21 | 12 | -1.75 |
| BTC Daily | transformer | Transformer | 263 | 107 | 156 | 40.68% | 40.00% | 40.68% | 9.32 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 263 | 98 | 165 | 37.26% | 37.08% | 37.26% | 12.74 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 273 | 99 | 174 | 36.26% | 36.67% | 36.26% | 13.74 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 263 | 93 | 170 | 35.36% | 35.83% | 35.36% | 14.64 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 261 | 137 | 124 | 52.49% | 52.50% | 52.49% | 2.49 pp | 13 | 21 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 261 | 123 | 138 | 47.13% | 47.92% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | transformer | Transformer | 261 | 123 | 138 | 47.13% | 47.50% | 47.13% | 2.87 pp | -15 | 21 | -0.71 |
| BTC Market Hours | xgb | XGBoost | 261 | 120 | 141 | 45.98% | 44.17% | 45.98% | 4.02 pp | -21 | 21 | -1.00 |
| BTC Market Hours | rf | RandomForest | 261 | 116 | 145 | 44.44% | 43.33% | 44.44% | 5.56 pp | -29 | 21 | -1.38 |
| BTC Market Hours | lstm | LSTM | 261 | 106 | 155 | 40.61% | 42.08% | 40.61% | 9.39 pp | -49 | 21 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 261 | 130 | 131 | 49.81% | 49.58% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 261 | 128 | 133 | 49.04% | 48.75% | 49.04% | 0.96 pp | -5 | 21 | -0.24 |
| BTC Market Hours Daily | nn | NN | 261 | 125 | 136 | 47.89% | 48.33% | 47.89% | 2.11 pp | -11 | 21 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 261 | 115 | 146 | 44.06% | 42.92% | 44.06% | 5.94 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | rf | RandomForest | 261 | 113 | 148 | 43.30% | 42.50% | 43.30% | 6.70 pp | -35 | 21 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 261 | 108 | 153 | 41.38% | 42.08% | 41.38% | 8.62 pp | -45 | 21 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 111 | 116 | 48.90% | 48.90% | 48.90% | 1.10 pp | -5 | 15 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 98 | 129 | 43.17% | 43.17% | 43.17% | 6.83 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 97 | 130 | 42.73% | 42.73% | 42.73% | 7.27 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 15 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
