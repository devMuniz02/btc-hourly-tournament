# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T17:42:35.303556+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 315 | 255 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 456 | 243 | 213 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 16:00:00+00:00 | 456 | 243 | 213 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 211 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 15:00:00+00:00 | 211 | 70 | 141 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 243 | 124 | 119 | 51.03% | 51.67% | 51.03% | 1.03 pp | 5 | 19 | 0.26 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 243 | 119 | 124 | 48.97% | 49.17% | 48.97% | 1.03 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 243 | 119 | 124 | 48.97% | 49.17% | 48.97% | 1.03 pp | -5 | 20 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 14 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 219 | 107 | 112 | 48.86% | 48.86% | 48.86% | 1.14 pp | -5 | 10 | -0.50 |
| BTC Market Hours Daily | nn | NN | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 20 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 243 | 115 | 128 | 47.33% | 47.50% | 47.33% | 2.67 pp | -13 | 19 | -0.68 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 211 | 100 | 111 | 47.39% | 47.39% | 47.39% | 2.61 pp | -11 | 14 | -0.79 |
| BTC Market Hours | transformer | Transformer | 243 | 113 | 130 | 46.50% | 46.67% | 46.50% | 3.50 pp | -17 | 19 | -0.89 |
| BTC Market Hours | xgb | XGBoost | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 19 | -0.89 |
| Consolidated Market Hours | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 70 | 32 | 38 | 45.71% | 45.71% | 45.71% | 4.29 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 211 | 98 | 113 | 46.45% | 46.45% | 46.45% | 3.55 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 243 | 110 | 133 | 45.27% | 45.42% | 45.27% | 4.73 pp | -23 | 19 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 243 | 108 | 135 | 44.44% | 44.17% | 44.44% | 5.56 pp | -27 | 20 | -1.35 |
| BTC Daily | mlp_sklearn | MLPClassifier | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 243 | 107 | 136 | 44.03% | 43.75% | 44.03% | 5.97 pp | -29 | 20 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 243 | 103 | 140 | 42.39% | 42.50% | 42.39% | 7.61 pp | -37 | 19 | -1.95 |
| Consolidated Market Hours | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 70 | 29 | 41 | 41.43% | 41.43% | 41.43% | 8.57 pp | -12 | 6 | -2.00 |
| BTC Daily | nn | NN | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 243 | 99 | 144 | 40.74% | 41.25% | 40.74% | 9.26 pp | -45 | 20 | -2.25 |
| BTC Hourly | transformer | Transformer | 219 | 98 | 121 | 44.75% | 44.75% | 44.75% | 5.25 pp | -23 | 10 | -2.30 |
| Consolidated Hourly | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 14 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 70 | 27 | 43 | 38.57% | 38.57% | 38.57% | 11.43 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 6 | -3.00 |
| BTC Hourly | nn | NN | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 219 | 92 | 127 | 42.01% | 42.01% | 42.01% | 7.99 pp | -35 | 10 | -3.50 |
| BTC Daily | transformer | Transformer | 245 | 99 | 146 | 40.41% | 40.00% | 40.41% | 9.59 pp | -47 | 11 | -4.27 |
| BTC Daily | rf | RandomForest | 245 | 94 | 151 | 38.37% | 37.92% | 38.37% | 11.63 pp | -57 | 11 | -5.18 |
| BTC Hourly | lstm | LSTM | 219 | 81 | 138 | 36.99% | 36.99% | 36.99% | 13.01 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 255 | 91 | 164 | 35.69% | 35.83% | 35.69% | 14.31 pp | -73 | 12 | -6.08 |
| BTC Hourly | xgb | XGBoost | 219 | 76 | 143 | 34.70% | 34.70% | 34.70% | 15.30 pp | -67 | 10 | -6.70 |
| BTC Daily | lstm | LSTM | 245 | 83 | 162 | 33.88% | 34.17% | 33.88% | 16.12 pp | -79 | 11 | -7.18 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 245 | 111 | 134 | 45.31% | 45.00% | 45.31% | 4.69 pp | -23 | 11 | -2.09 |
| BTC Daily | transformer | Transformer | 245 | 99 | 146 | 40.41% | 40.00% | 40.41% | 9.59 pp | -47 | 11 | -4.27 |
| BTC Daily | rf | RandomForest | 245 | 94 | 151 | 38.37% | 37.92% | 38.37% | 11.63 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 255 | 91 | 164 | 35.69% | 35.83% | 35.69% | 14.31 pp | -73 | 12 | -6.08 |
| BTC Daily | lstm | LSTM | 245 | 83 | 162 | 33.88% | 34.17% | 33.88% | 16.12 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 243 | 124 | 119 | 51.03% | 51.67% | 51.03% | 1.03 pp | 5 | 19 | 0.26 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 243 | 115 | 128 | 47.33% | 47.50% | 47.33% | 2.67 pp | -13 | 19 | -0.68 |
| BTC Market Hours | transformer | Transformer | 243 | 113 | 130 | 46.50% | 46.67% | 46.50% | 3.50 pp | -17 | 19 | -0.89 |
| BTC Market Hours | xgb | XGBoost | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 19 | -0.89 |
| BTC Market Hours | rf | RandomForest | 243 | 110 | 133 | 45.27% | 45.42% | 45.27% | 4.73 pp | -23 | 19 | -1.21 |
| BTC Market Hours | lstm | LSTM | 243 | 103 | 140 | 42.39% | 42.50% | 42.39% | 7.61 pp | -37 | 19 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 243 | 119 | 124 | 48.97% | 49.17% | 48.97% | 1.03 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 243 | 119 | 124 | 48.97% | 49.17% | 48.97% | 1.03 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | nn | NN | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 20 | -0.55 |
| BTC Market Hours Daily | xgb | XGBoost | 243 | 108 | 135 | 44.44% | 44.17% | 44.44% | 5.56 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | rf | RandomForest | 243 | 107 | 136 | 44.03% | 43.75% | 44.03% | 5.97 pp | -29 | 20 | -1.45 |
| BTC Market Hours Daily | lstm | LSTM | 243 | 99 | 144 | 40.74% | 41.25% | 40.74% | 9.26 pp | -45 | 20 | -2.25 |

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
