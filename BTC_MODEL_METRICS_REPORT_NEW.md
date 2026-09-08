# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T01:29:39.259579+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 268 | 208 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 304 | 244 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 440 | 232 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 440 | 232 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 202 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T23:00:00+00:00 | 203 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 232 | 120 | 112 | 51.72% | 51.72% | 51.72% | 1.72 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | transformer | Transformer | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 19 | -0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 232 | 114 | 118 | 49.14% | 49.14% | 49.14% | 0.86 pp | -4 | 19 | -0.21 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 9 | -0.22 |
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 19 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 18 | -0.56 |
| Consolidated Market Hours | xgb | XGBoost | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| BTC Market Hours | transformer | Transformer | 232 | 109 | 123 | 46.98% | 46.98% | 46.98% | 3.02 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 232 | 106 | 126 | 45.69% | 45.69% | 45.69% | 4.31 pp | -20 | 18 | -1.11 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 30 | 36 | 45.45% | 45.45% | 45.45% | 4.55 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 19 | -1.37 |
| Consolidated Market Hours | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 11 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 232 | 100 | 132 | 43.10% | 43.10% | 43.10% | 6.90 pp | -32 | 19 | -1.68 |
| Consolidated Market Hours | lstm | LSTM | 65 | 28 | 37 | 43.08% | 43.08% | 43.08% | 6.92 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 18 | -2.00 |
| BTC Daily | nn | NN | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 11 | -2.36 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 232 | 93 | 139 | 40.09% | 40.09% | 40.09% | 9.91 pp | -46 | 19 | -2.42 |
| BTC Hourly | transformer | Transformer | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 208 | 88 | 120 | 42.31% | 42.31% | 42.31% | 7.69 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 9 | -4.00 |
| BTC Daily | transformer | Transformer | 234 | 94 | 140 | 40.17% | 40.17% | 40.17% | 9.83 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 11 | -4.91 |
| BTC Hourly | lstm | LSTM | 208 | 77 | 131 | 37.02% | 37.02% | 37.02% | 12.98 pp | -54 | 9 | -6.00 |
| BTC Daily | xgb | XGBoost | 244 | 86 | 158 | 35.25% | 35.42% | 35.25% | 14.75 pp | -72 | 12 | -6.00 |
| BTC Daily | lstm | LSTM | 234 | 78 | 156 | 33.33% | 33.33% | 33.33% | 16.67 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 208 | 71 | 137 | 34.13% | 34.13% | 34.13% | 15.87 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 9 | -0.22 |
| BTC Hourly | transformer | Transformer | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 208 | 88 | 120 | 42.31% | 42.31% | 42.31% | 7.69 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 9 | -4.00 |
| BTC Hourly | lstm | LSTM | 208 | 77 | 131 | 37.02% | 37.02% | 37.02% | 12.98 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 208 | 71 | 137 | 34.13% | 34.13% | 34.13% | 15.87 pp | -66 | 9 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 234 | 109 | 125 | 46.58% | 46.58% | 46.58% | 3.42 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 234 | 104 | 130 | 44.44% | 44.44% | 44.44% | 5.56 pp | -26 | 11 | -2.36 |
| BTC Daily | transformer | Transformer | 234 | 94 | 140 | 40.17% | 40.17% | 40.17% | 9.83 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 11 | -4.91 |
| BTC Daily | xgb | XGBoost | 244 | 86 | 158 | 35.25% | 35.42% | 35.25% | 14.75 pp | -72 | 12 | -6.00 |
| BTC Daily | lstm | LSTM | 234 | 78 | 156 | 33.33% | 33.33% | 33.33% | 16.67 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 232 | 120 | 112 | 51.72% | 51.72% | 51.72% | 1.72 pp | 8 | 18 | 0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 18 | -0.56 |
| BTC Market Hours | transformer | Transformer | 232 | 109 | 123 | 46.98% | 46.98% | 46.98% | 3.02 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 232 | 106 | 126 | 45.69% | 45.69% | 45.69% | 4.31 pp | -20 | 18 | -1.11 |
| BTC Market Hours | lstm | LSTM | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 18 | -2.00 |

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
| Consolidated Hourly | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 202 | 97 | 105 | 48.02% | 48.02% | 48.02% | 1.98 pp | -8 | 13 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 202 | 93 | 109 | 46.04% | 46.04% | 46.04% | 3.96 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 13 | -2.00 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 66 | 32 | 34 | 48.48% | 48.48% | 48.48% | 1.52 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 66 | 30 | 36 | 45.45% | 45.45% | 45.45% | 4.55 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
