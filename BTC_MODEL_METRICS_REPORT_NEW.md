# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T12:59:03.405394+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 275 | 215 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 311 | 251 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 447 | 239 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 447 | 239 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 207 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 13:00:00+00:00 | 207 | 68 | 139 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 239 | 121 | 118 | 50.63% | 50.63% | 50.63% | 0.63 pp | 3 | 19 | 0.16 |
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| BTC Market Hours Daily | transformer | Transformer | 239 | 118 | 121 | 49.37% | 49.37% | 49.37% | 0.63 pp | -3 | 20 | -0.15 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 239 | 116 | 123 | 48.54% | 48.54% | 48.54% | 1.46 pp | -7 | 20 | -0.35 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 215 | 105 | 110 | 48.84% | 48.84% | 48.84% | 1.16 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | nn | NN | 239 | 113 | 126 | 47.28% | 47.28% | 47.28% | 2.72 pp | -13 | 20 | -0.65 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 19 | -0.79 |
| BTC Market Hours | transformer | Transformer | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 19 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 239 | 111 | 128 | 46.44% | 46.44% | 46.44% | 3.56 pp | -17 | 19 | -0.89 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 19 | -1.21 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 241 | 113 | 128 | 46.89% | 46.67% | 46.89% | 3.11 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 239 | 105 | 134 | 43.93% | 43.93% | 43.93% | 6.07 pp | -29 | 20 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 239 | 105 | 134 | 43.93% | 43.93% | 43.93% | 6.07 pp | -29 | 20 | -1.45 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| BTC Daily | nn | NN | 241 | 110 | 131 | 45.64% | 45.42% | 45.64% | 4.36 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 239 | 101 | 138 | 42.26% | 42.26% | 42.26% | 7.74 pp | -37 | 19 | -1.95 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 239 | 97 | 142 | 40.59% | 40.59% | 40.59% | 9.41 pp | -45 | 20 | -2.25 |
| BTC Hourly | transformer | Transformer | 215 | 97 | 118 | 45.12% | 45.12% | 45.12% | 4.88 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| BTC Hourly | nn | NN | 215 | 91 | 124 | 42.33% | 42.33% | 42.33% | 7.67 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 241 | 98 | 143 | 40.66% | 40.42% | 40.66% | 9.34 pp | -45 | 11 | -4.09 |
| BTC Hourly | rf | RandomForest | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 9 | -4.11 |
| BTC Daily | rf | RandomForest | 241 | 92 | 149 | 38.17% | 37.92% | 38.17% | 11.83 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 251 | 90 | 161 | 35.86% | 35.83% | 35.86% | 14.14 pp | -71 | 12 | -5.92 |
| BTC Hourly | lstm | LSTM | 215 | 80 | 135 | 37.21% | 37.21% | 37.21% | 12.79 pp | -55 | 9 | -6.11 |
| BTC Daily | lstm | LSTM | 241 | 81 | 160 | 33.61% | 33.75% | 33.61% | 16.39 pp | -79 | 11 | -7.18 |
| BTC Hourly | xgb | XGBoost | 215 | 74 | 141 | 34.42% | 34.42% | 34.42% | 15.58 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 215 | 105 | 110 | 48.84% | 48.84% | 48.84% | 1.16 pp | -5 | 9 | -0.56 |
| BTC Hourly | transformer | Transformer | 215 | 97 | 118 | 45.12% | 45.12% | 45.12% | 4.88 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 215 | 91 | 124 | 42.33% | 42.33% | 42.33% | 7.67 pp | -33 | 9 | -3.67 |
| BTC Hourly | rf | RandomForest | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 215 | 80 | 135 | 37.21% | 37.21% | 37.21% | 12.79 pp | -55 | 9 | -6.11 |
| BTC Hourly | xgb | XGBoost | 215 | 74 | 141 | 34.42% | 34.42% | 34.42% | 15.58 pp | -67 | 9 | -7.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 241 | 113 | 128 | 46.89% | 46.67% | 46.89% | 3.11 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 241 | 110 | 131 | 45.64% | 45.42% | 45.64% | 4.36 pp | -21 | 11 | -1.91 |
| BTC Daily | transformer | Transformer | 241 | 98 | 143 | 40.66% | 40.42% | 40.66% | 9.34 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 241 | 92 | 149 | 38.17% | 37.92% | 38.17% | 11.83 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 251 | 90 | 161 | 35.86% | 35.83% | 35.86% | 14.14 pp | -71 | 12 | -5.92 |
| BTC Daily | lstm | LSTM | 241 | 81 | 160 | 33.61% | 33.75% | 33.61% | 16.39 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 239 | 121 | 118 | 50.63% | 50.63% | 50.63% | 0.63 pp | 3 | 19 | 0.16 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 19 | -0.79 |
| BTC Market Hours | transformer | Transformer | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 19 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 239 | 111 | 128 | 46.44% | 46.44% | 46.44% | 3.56 pp | -17 | 19 | -0.89 |
| BTC Market Hours | rf | RandomForest | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 19 | -1.21 |
| BTC Market Hours | lstm | LSTM | 239 | 101 | 138 | 42.26% | 42.26% | 42.26% | 7.74 pp | -37 | 19 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 239 | 118 | 121 | 49.37% | 49.37% | 49.37% | 0.63 pp | -3 | 20 | -0.15 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 239 | 116 | 123 | 48.54% | 48.54% | 48.54% | 1.46 pp | -7 | 20 | -0.35 |
| BTC Market Hours Daily | nn | NN | 239 | 113 | 126 | 47.28% | 47.28% | 47.28% | 2.72 pp | -13 | 20 | -0.65 |
| BTC Market Hours Daily | rf | RandomForest | 239 | 105 | 134 | 43.93% | 43.93% | 43.93% | 6.07 pp | -29 | 20 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 239 | 105 | 134 | 43.93% | 43.93% | 43.93% | 6.07 pp | -29 | 20 | -1.45 |
| BTC Market Hours Daily | lstm | LSTM | 239 | 97 | 142 | 40.59% | 40.59% | 40.59% | 9.41 pp | -45 | 20 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 103 | 104 | 49.76% | 49.76% | 49.76% | 0.24 pp | -1 | 14 | -0.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 91 | 116 | 43.96% | 43.96% | 43.96% | 6.04 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 88 | 119 | 42.51% | 42.51% | 42.51% | 7.49 pp | -31 | 14 | -2.21 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
