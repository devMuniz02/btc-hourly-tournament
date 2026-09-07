# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T20:24:46.242629+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 265 | 205 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 300 | 240 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 19:00:00+00:00 | 431 | 228 | 203 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 19:00:00+00:00 | 431 | 228 | 203 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 228 | 119 | 109 | 52.19% | 52.19% | 52.19% | 2.19 pp | 10 | 18 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 228 | 114 | 114 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 228 | 111 | 117 | 48.68% | 48.68% | 48.68% | 1.32 pp | -6 | 19 | -0.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 19 | -0.63 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | transformer | Transformer | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 228 | 105 | 123 | 46.05% | 46.05% | 46.05% | 3.95 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 228 | 101 | 127 | 44.30% | 44.30% | 44.30% | 5.70 pp | -26 | 19 | -1.37 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 10 | -1.80 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 228 | 96 | 132 | 42.11% | 42.11% | 42.11% | 7.89 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 18 | -2.22 |
| BTC Hourly | transformer | Transformer | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 228 | 91 | 137 | 39.91% | 39.91% | 39.91% | 10.09 pp | -46 | 19 | -2.42 |
| BTC Daily | nn | NN | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 205 | 87 | 118 | 42.44% | 42.44% | 42.44% | 7.56 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 205 | 85 | 120 | 41.46% | 41.46% | 41.46% | 8.54 pp | -35 | 9 | -3.89 |
| BTC Daily | transformer | Transformer | 230 | 91 | 139 | 39.57% | 39.57% | 39.57% | 10.43 pp | -48 | 10 | -4.80 |
| BTC Daily | rf | RandomForest | 230 | 87 | 143 | 37.83% | 37.83% | 37.83% | 12.17 pp | -56 | 10 | -5.60 |
| BTC Hourly | lstm | LSTM | 205 | 76 | 129 | 37.07% | 37.07% | 37.07% | 12.93 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 205 | 71 | 134 | 34.63% | 34.63% | 34.63% | 15.37 pp | -63 | 9 | -7.00 |
| BTC Daily | lstm | LSTM | 230 | 77 | 153 | 33.48% | 33.48% | 33.48% | 16.52 pp | -76 | 10 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 9 | 0.11 |
| BTC Hourly | transformer | Transformer | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 205 | 87 | 118 | 42.44% | 42.44% | 42.44% | 7.56 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 205 | 85 | 120 | 41.46% | 41.46% | 41.46% | 8.54 pp | -35 | 9 | -3.89 |
| BTC Hourly | lstm | LSTM | 205 | 76 | 129 | 37.07% | 37.07% | 37.07% | 12.93 pp | -53 | 9 | -5.89 |
| BTC Hourly | xgb | XGBoost | 205 | 71 | 134 | 34.63% | 34.63% | 34.63% | 15.37 pp | -63 | 9 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 10 | -1.80 |
| BTC Daily | nn | NN | 230 | 102 | 128 | 44.35% | 44.35% | 44.35% | 5.65 pp | -26 | 10 | -2.60 |
| BTC Daily | transformer | Transformer | 230 | 91 | 139 | 39.57% | 39.57% | 39.57% | 10.43 pp | -48 | 10 | -4.80 |
| BTC Daily | rf | RandomForest | 230 | 87 | 143 | 37.83% | 37.83% | 37.83% | 12.17 pp | -56 | 10 | -5.60 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 230 | 77 | 153 | 33.48% | 33.48% | 33.48% | 16.52 pp | -76 | 10 | -7.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 228 | 119 | 109 | 52.19% | 52.19% | 52.19% | 2.19 pp | 10 | 18 | 0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | transformer | Transformer | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 228 | 105 | 123 | 46.05% | 46.05% | 46.05% | 3.95 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 18 | -1.33 |
| BTC Market Hours | lstm | LSTM | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 18 | -2.22 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 228 | 114 | 114 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 228 | 111 | 117 | 48.68% | 48.68% | 48.68% | 1.32 pp | -6 | 19 | -0.32 |
| BTC Market Hours Daily | nn | NN | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 19 | -0.63 |
| BTC Market Hours Daily | rf | RandomForest | 228 | 101 | 127 | 44.30% | 44.30% | 44.30% | 5.70 pp | -26 | 19 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 228 | 96 | 132 | 42.11% | 42.11% | 42.11% | 7.89 pp | -36 | 19 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 228 | 91 | 137 | 39.91% | 39.91% | 39.91% | 10.09 pp | -46 | 19 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
