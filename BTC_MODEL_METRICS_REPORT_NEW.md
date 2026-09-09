# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T14:29:43.207577+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 293 | 233 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 328 | 268 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 13:00:00+00:00 | 479 | 256 | 223 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 13:00:00+00:00 | 479 | 256 | 223 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.83% | 46.12% | 3.88 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 258 | 116 | 142 | 44.96% | 44.17% | 44.96% | 5.04 pp | -26 | 12 | -2.17 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Hourly | transformer | Transformer | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 10 | -2.70 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Hourly | nn | NN | 233 | 98 | 135 | 42.06% | 42.06% | 42.06% | 7.94 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 10 | -4.10 |
| BTC Daily | transformer | Transformer | 258 | 103 | 155 | 39.92% | 39.17% | 39.92% | 10.08 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 258 | 96 | 162 | 37.21% | 36.67% | 37.21% | 12.79 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.42% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Hourly | lstm | LSTM | 233 | 87 | 146 | 37.34% | 37.34% | 37.34% | 12.66 pp | -59 | 10 | -5.90 |
| BTC Daily | lstm | LSTM | 258 | 91 | 167 | 35.27% | 35.83% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 233 | 80 | 153 | 34.33% | 34.33% | 34.33% | 15.67 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| BTC Hourly | transformer | Transformer | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 233 | 98 | 135 | 42.06% | 42.06% | 42.06% | 7.94 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 10 | -4.10 |
| BTC Hourly | lstm | LSTM | 233 | 87 | 146 | 37.34% | 37.34% | 37.34% | 12.66 pp | -59 | 10 | -5.90 |
| BTC Hourly | xgb | XGBoost | 233 | 80 | 153 | 34.33% | 34.33% | 34.33% | 15.67 pp | -73 | 10 | -7.30 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.83% | 46.12% | 3.88 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 258 | 116 | 142 | 44.96% | 44.17% | 44.96% | 5.04 pp | -26 | 12 | -2.17 |
| BTC Daily | transformer | Transformer | 258 | 103 | 155 | 39.92% | 39.17% | 39.92% | 10.08 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 258 | 96 | 162 | 37.21% | 36.67% | 37.21% | 12.79 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.42% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 258 | 91 | 167 | 35.27% | 35.83% | 35.27% | 14.73 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
