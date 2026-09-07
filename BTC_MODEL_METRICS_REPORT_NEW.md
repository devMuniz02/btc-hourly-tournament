# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T20:15:10.247665+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 264 | 204 | 60 | 0 |
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
| BTC Hourly | mlp_sklearn | MLPClassifier | 204 | 103 | 101 | 50.49% | 50.49% | 50.49% | 0.49 pp | 2 | 9 | 0.22 |
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
| BTC Hourly | transformer | Transformer | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Market Hours | lstm | LSTM | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 18 | -2.22 |
| BTC Daily | nn | NN | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 228 | 91 | 137 | 39.91% | 39.91% | 39.91% | 10.09 pp | -46 | 19 | -2.42 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 204 | 87 | 117 | 42.65% | 42.65% | 42.65% | 7.35 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 230 | 92 | 138 | 40.00% | 40.00% | 40.00% | 10.00 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 230 | 88 | 142 | 38.26% | 38.26% | 38.26% | 11.74 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 204 | 76 | 128 | 37.25% | 37.25% | 37.25% | 12.75 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 204 | 71 | 133 | 34.80% | 34.80% | 34.80% | 15.20 pp | -62 | 9 | -6.89 |
| BTC Daily | lstm | LSTM | 230 | 76 | 154 | 33.04% | 33.04% | 33.04% | 16.96 pp | -78 | 10 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 204 | 103 | 101 | 50.49% | 50.49% | 50.49% | 0.49 pp | 2 | 9 | 0.22 |
| BTC Hourly | transformer | Transformer | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Hourly | nn | NN | 204 | 87 | 117 | 42.65% | 42.65% | 42.65% | 7.35 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 9 | -3.78 |
| BTC Hourly | lstm | LSTM | 204 | 76 | 128 | 37.25% | 37.25% | 37.25% | 12.75 pp | -52 | 9 | -5.78 |
| BTC Hourly | xgb | XGBoost | 204 | 71 | 133 | 34.80% | 34.80% | 34.80% | 15.20 pp | -62 | 9 | -6.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 10 | -1.80 |
| BTC Daily | nn | NN | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 230 | 92 | 138 | 40.00% | 40.00% | 40.00% | 10.00 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 230 | 88 | 142 | 38.26% | 38.26% | 38.26% | 11.74 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 230 | 76 | 154 | 33.04% | 33.04% | 33.04% | 16.96 pp | -78 | 10 | -7.80 |

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
