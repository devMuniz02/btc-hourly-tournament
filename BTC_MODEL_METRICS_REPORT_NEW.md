# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T02:04:27.705318+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 252 | 192 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 288 | 228 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 411 | 216 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 411 | 216 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 186 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T14:00:00+00:00 | 187 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 192 | 98 | 94 | 51.04% | 51.04% | 51.04% | 1.04 pp | 4 | 8 | 0.50 |
| BTC Market Hours | nn | NN | 216 | 112 | 104 | 51.85% | 51.85% | 51.85% | 1.85 pp | 8 | 17 | 0.47 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | nn | NN | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 18 | -0.33 |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | transformer | Transformer | 216 | 104 | 112 | 48.15% | 48.15% | 48.15% | 1.85 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 17 | -0.59 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| BTC Market Hours | transformer | Transformer | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| Consolidated Hourly | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 18 | -1.22 |
| Consolidated Hourly | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 10 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 216 | 92 | 124 | 42.59% | 42.59% | 42.59% | 7.41 pp | -32 | 17 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 216 | 90 | 126 | 41.67% | 41.67% | 41.67% | 8.33 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Daily | nn | NN | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 8 | -2.25 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 8 | -3.50 |
| BTC Market Hours | lstm | LSTM | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 17 | -3.65 |
| BTC Market Hours Daily | lstm | LSTM | 216 | 75 | 141 | 34.72% | 34.72% | 34.72% | 15.28 pp | -66 | 18 | -3.67 |
| BTC Hourly | rf | RandomForest | 192 | 80 | 112 | 41.67% | 41.67% | 41.67% | 8.33 pp | -32 | 8 | -4.00 |
| BTC Daily | transformer | Transformer | 218 | 89 | 129 | 40.83% | 40.83% | 40.83% | 9.17 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 218 | 84 | 134 | 38.53% | 38.53% | 38.53% | 11.47 pp | -50 | 10 | -5.00 |
| BTC Hourly | lstm | LSTM | 192 | 72 | 120 | 37.50% | 37.50% | 37.50% | 12.50 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 228 | 81 | 147 | 35.53% | 35.53% | 35.53% | 14.47 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 8 | -6.50 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 192 | 98 | 94 | 51.04% | 51.04% | 51.04% | 1.04 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 8 | -2.25 |
| BTC Hourly | nn | NN | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 192 | 80 | 112 | 41.67% | 41.67% | 41.67% | 8.33 pp | -32 | 8 | -4.00 |
| BTC Hourly | lstm | LSTM | 192 | 72 | 120 | 37.50% | 37.50% | 37.50% | 12.50 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 8 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 102 | 116 | 46.79% | 46.79% | 46.79% | 3.21 pp | -14 | 10 | -1.40 |
| BTC Daily | nn | NN | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 218 | 89 | 129 | 40.83% | 40.83% | 40.83% | 9.17 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 218 | 84 | 134 | 38.53% | 38.53% | 38.53% | 11.47 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 228 | 81 | 147 | 35.53% | 35.53% | 35.53% | 14.47 pp | -66 | 11 | -6.00 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 216 | 112 | 104 | 51.85% | 51.85% | 51.85% | 1.85 pp | 8 | 17 | 0.47 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| BTC Market Hours | transformer | Transformer | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 216 | 92 | 124 | 42.59% | 42.59% | 42.59% | 7.41 pp | -32 | 17 | -1.88 |
| BTC Market Hours | lstm | LSTM | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 17 | -3.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 216 | 104 | 112 | 48.15% | 48.15% | 48.15% | 1.85 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 18 | -1.22 |
| BTC Market Hours Daily | xgb | XGBoost | 216 | 90 | 126 | 41.67% | 41.67% | 41.67% | 8.33 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 216 | 75 | 141 | 34.72% | 34.72% | 34.72% | 15.28 pp | -66 | 18 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 13 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 186 | 91 | 95 | 48.92% | 48.92% | 48.92% | 1.08 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 186 | 84 | 102 | 45.16% | 45.16% | 45.16% | 4.84 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 13 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
