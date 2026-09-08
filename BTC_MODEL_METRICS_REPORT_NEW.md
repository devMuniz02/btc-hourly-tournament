# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T17:22:29.431045+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 279 | 219 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 314 | 254 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 455 | 242 | 213 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 455 | 242 | 213 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 242 | 124 | 118 | 51.24% | 51.67% | 51.24% | 1.24 pp | 6 | 19 | 0.32 |
| BTC Market Hours Daily | transformer | Transformer | 242 | 119 | 123 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 48.75% | 48.76% | 1.24 pp | -6 | 20 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 219 | 107 | 112 | 48.86% | 48.86% | 48.86% | 1.14 pp | -5 | 10 | -0.50 |
| BTC Market Hours Daily | nn | NN | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 19 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| BTC Market Hours | transformer | Transformer | 242 | 113 | 129 | 46.69% | 47.08% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 242 | 110 | 132 | 45.45% | 45.83% | 45.45% | 4.55 pp | -22 | 19 | -1.16 |
| BTC Market Hours Daily | xgb | XGBoost | 242 | 108 | 134 | 44.63% | 44.17% | 44.63% | 5.37 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 242 | 107 | 135 | 44.21% | 43.75% | 44.21% | 5.79 pp | -28 | 20 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 244 | 114 | 130 | 46.72% | 47.08% | 46.72% | 3.28 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 242 | 103 | 139 | 42.56% | 42.92% | 42.56% | 7.44 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Daily | nn | NN | 244 | 110 | 134 | 45.08% | 45.00% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Hourly | transformer | Transformer | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 242 | 98 | 144 | 40.50% | 40.83% | 40.50% | 9.50 pp | -46 | 20 | -2.30 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Daily | transformer | Transformer | 244 | 98 | 146 | 40.16% | 40.00% | 40.16% | 9.84 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 244 | 93 | 151 | 38.11% | 37.92% | 38.11% | 11.89 pp | -58 | 11 | -5.27 |
| BTC Hourly | lstm | LSTM | 219 | 81 | 138 | 36.99% | 36.99% | 36.99% | 13.01 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 254 | 90 | 164 | 35.43% | 35.83% | 35.43% | 14.57 pp | -74 | 12 | -6.17 |
| BTC Hourly | xgb | XGBoost | 219 | 76 | 143 | 34.70% | 34.70% | 34.70% | 15.30 pp | -67 | 10 | -6.70 |
| BTC Daily | lstm | LSTM | 244 | 83 | 161 | 34.02% | 34.17% | 34.02% | 15.98 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 219 | 107 | 112 | 48.86% | 48.86% | 48.86% | 1.14 pp | -5 | 10 | -0.50 |
| BTC Hourly | transformer | Transformer | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| BTC Hourly | nn | NN | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Hourly | lstm | LSTM | 219 | 81 | 138 | 36.99% | 36.99% | 36.99% | 13.01 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 219 | 76 | 143 | 34.70% | 34.70% | 34.70% | 15.30 pp | -67 | 10 | -6.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 244 | 114 | 130 | 46.72% | 47.08% | 46.72% | 3.28 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 244 | 110 | 134 | 45.08% | 45.00% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 244 | 98 | 146 | 40.16% | 40.00% | 40.16% | 9.84 pp | -48 | 11 | -4.36 |
| BTC Daily | rf | RandomForest | 244 | 93 | 151 | 38.11% | 37.92% | 38.11% | 11.89 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 254 | 90 | 164 | 35.43% | 35.83% | 35.43% | 14.57 pp | -74 | 12 | -6.17 |
| BTC Daily | lstm | LSTM | 244 | 83 | 161 | 34.02% | 34.17% | 34.02% | 15.98 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 242 | 124 | 118 | 51.24% | 51.67% | 51.24% | 1.24 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 242 | 113 | 129 | 46.69% | 47.08% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 242 | 110 | 132 | 45.45% | 45.83% | 45.45% | 4.55 pp | -22 | 19 | -1.16 |
| BTC Market Hours | lstm | LSTM | 242 | 103 | 139 | 42.56% | 42.92% | 42.56% | 7.44 pp | -36 | 19 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 242 | 119 | 123 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 48.75% | 48.76% | 1.24 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 242 | 108 | 134 | 44.63% | 44.17% | 44.63% | 5.37 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 242 | 107 | 135 | 44.21% | 43.75% | 44.21% | 5.79 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 242 | 98 | 144 | 40.50% | 40.83% | 40.50% | 9.50 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
