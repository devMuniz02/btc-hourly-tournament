# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T02:29:31.040408+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 269 | 209 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 305 | 245 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 441 | 233 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 440 | 232 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 202 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 23:00:00+00:00 | 202 | 65 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 233 | 120 | 113 | 51.50% | 51.50% | 51.50% | 1.50 pp | 7 | 18 | 0.39 |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 209 | 104 | 105 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 232 | 114 | 118 | 49.14% | 49.14% | 49.14% | 0.86 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | nn | NN | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 19 | -0.53 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 18 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 18 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 18 | -1.06 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 19 | -1.37 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | xgb | XGBoost | 232 | 100 | 132 | 43.10% | 43.10% | 43.10% | 6.90 pp | -32 | 19 | -1.68 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| BTC Market Hours | lstm | LSTM | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 18 | -1.94 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |
| BTC Daily | nn | NN | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 11 | -2.27 |
| BTC Hourly | transformer | Transformer | 209 | 94 | 115 | 44.98% | 44.98% | 44.98% | 5.02 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 19 | -2.42 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 235 | 95 | 140 | 40.43% | 40.43% | 40.43% | 9.57 pp | -45 | 11 | -4.09 |
| BTC Hourly | rf | RandomForest | 209 | 86 | 123 | 41.15% | 41.15% | 41.15% | 8.85 pp | -37 | 9 | -4.11 |
| BTC Daily | rf | RandomForest | 235 | 90 | 145 | 38.30% | 38.30% | 38.30% | 11.70 pp | -55 | 11 | -5.00 |
| BTC Daily | xgb | XGBoost | 245 | 87 | 158 | 35.51% | 35.83% | 35.51% | 14.49 pp | -71 | 12 | -5.92 |
| BTC Hourly | lstm | LSTM | 209 | 77 | 132 | 36.84% | 36.84% | 36.84% | 13.16 pp | -55 | 9 | -6.11 |
| BTC Daily | lstm | LSTM | 235 | 79 | 156 | 33.62% | 33.62% | 33.62% | 16.38 pp | -77 | 11 | -7.00 |
| BTC Hourly | xgb | XGBoost | 209 | 71 | 138 | 33.97% | 33.97% | 33.97% | 16.03 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 209 | 104 | 105 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 209 | 94 | 115 | 44.98% | 44.98% | 44.98% | 5.02 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 209 | 86 | 123 | 41.15% | 41.15% | 41.15% | 8.85 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 209 | 77 | 132 | 36.84% | 36.84% | 36.84% | 13.16 pp | -55 | 9 | -6.11 |
| BTC Hourly | xgb | XGBoost | 209 | 71 | 138 | 33.97% | 33.97% | 33.97% | 16.03 pp | -67 | 9 | -7.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 235 | 105 | 130 | 44.68% | 44.68% | 44.68% | 5.32 pp | -25 | 11 | -2.27 |
| BTC Daily | transformer | Transformer | 235 | 95 | 140 | 40.43% | 40.43% | 40.43% | 9.57 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 235 | 90 | 145 | 38.30% | 38.30% | 38.30% | 11.70 pp | -55 | 11 | -5.00 |
| BTC Daily | xgb | XGBoost | 245 | 87 | 158 | 35.51% | 35.83% | 35.51% | 14.49 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 235 | 79 | 156 | 33.62% | 33.62% | 33.62% | 16.38 pp | -77 | 11 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 233 | 120 | 113 | 51.50% | 51.50% | 51.50% | 1.50 pp | 7 | 18 | 0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 233 | 111 | 122 | 47.64% | 47.64% | 47.64% | 2.36 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 233 | 109 | 124 | 46.78% | 46.78% | 46.78% | 3.22 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 18 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 233 | 107 | 126 | 45.92% | 45.92% | 45.92% | 4.08 pp | -19 | 18 | -1.06 |
| BTC Market Hours | lstm | LSTM | 233 | 99 | 134 | 42.49% | 42.49% | 42.49% | 7.51 pp | -35 | 18 | -1.94 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 232 | 114 | 118 | 49.14% | 49.14% | 49.14% | 0.86 pp | -4 | 19 | -0.21 |
| BTC Market Hours Daily | nn | NN | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 19 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 19 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 232 | 100 | 132 | 43.10% | 43.10% | 43.10% | 6.90 pp | -32 | 19 | -1.68 |
| BTC Market Hours Daily | lstm | LSTM | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 19 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Hourly | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 90 | 112 | 44.55% | 44.55% | 44.55% | 5.45 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 89 | 113 | 44.06% | 44.06% | 44.06% | 5.94 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 13 | -2.15 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
