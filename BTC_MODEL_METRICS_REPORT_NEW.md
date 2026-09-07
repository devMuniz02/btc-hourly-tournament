# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T21:47:36.350859+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 301 | 241 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 20:00:00+00:00 | 433 | 229 | 204 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 20:00:00+00:00 | 433 | 229 | 204 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 199 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 21:00:00+00:00 | 199 | 63 | 136 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 229 | 119 | 110 | 51.97% | 51.97% | 51.97% | 1.97 pp | 9 | 18 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 205 | 103 | 102 | 50.24% | 50.24% | 50.24% | 0.24 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 100 | 99 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 19 | -0.26 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 19 | -0.58 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 229 | 108 | 121 | 47.16% | 47.16% | 47.16% | 2.84 pp | -13 | 18 | -0.72 |
| BTC Market Hours | rf | RandomForest | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 18 | -0.94 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 229 | 103 | 126 | 44.98% | 44.98% | 44.98% | 5.02 pp | -23 | 18 | -1.28 |
| Consolidated Hourly | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 91 | 108 | 45.73% | 45.73% | 45.73% | 4.27 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 19 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 10 | -1.70 |
| Consolidated Hourly | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 88 | 111 | 44.22% | 44.22% | 44.22% | 5.78 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 19 | -1.84 |
| Consolidated Hourly | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 18 | -2.17 |
| BTC Hourly | transformer | Transformer | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 19 | -2.37 |
| BTC Daily | nn | NN | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 205 | 87 | 118 | 42.44% | 42.44% | 42.44% | 7.56 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 205 | 85 | 120 | 41.46% | 41.46% | 41.46% | 8.54 pp | -35 | 9 | -3.89 |
| BTC Daily | transformer | Transformer | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 10 | -4.70 |
| BTC Daily | rf | RandomForest | 231 | 88 | 143 | 38.10% | 38.10% | 38.10% | 11.90 pp | -55 | 10 | -5.50 |
| BTC Hourly | lstm | LSTM | 205 | 76 | 129 | 37.07% | 37.07% | 37.07% | 12.93 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 241 | 85 | 156 | 35.27% | 35.00% | 35.27% | 14.73 pp | -71 | 11 | -6.45 |
| BTC Hourly | xgb | XGBoost | 205 | 71 | 134 | 34.63% | 34.63% | 34.63% | 15.37 pp | -63 | 9 | -7.00 |
| BTC Daily | lstm | LSTM | 231 | 77 | 154 | 33.33% | 33.33% | 33.33% | 16.67 pp | -77 | 10 | -7.70 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 231 | 107 | 124 | 46.32% | 46.32% | 46.32% | 3.68 pp | -17 | 10 | -1.70 |
| BTC Daily | nn | NN | 231 | 103 | 128 | 44.59% | 44.59% | 44.59% | 5.41 pp | -25 | 10 | -2.50 |
| BTC Daily | transformer | Transformer | 231 | 92 | 139 | 39.83% | 39.83% | 39.83% | 10.17 pp | -47 | 10 | -4.70 |
| BTC Daily | rf | RandomForest | 231 | 88 | 143 | 38.10% | 38.10% | 38.10% | 11.90 pp | -55 | 10 | -5.50 |
| BTC Daily | xgb | XGBoost | 241 | 85 | 156 | 35.27% | 35.00% | 35.27% | 14.73 pp | -71 | 11 | -6.45 |
| BTC Daily | lstm | LSTM | 231 | 77 | 154 | 33.33% | 33.33% | 33.33% | 16.67 pp | -77 | 10 | -7.70 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 229 | 119 | 110 | 51.97% | 51.97% | 51.97% | 1.97 pp | 9 | 18 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 18 | -0.61 |
| BTC Market Hours | transformer | Transformer | 229 | 108 | 121 | 47.16% | 47.16% | 47.16% | 2.84 pp | -13 | 18 | -0.72 |
| BTC Market Hours | rf | RandomForest | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 18 | -0.94 |
| BTC Market Hours | xgb | XGBoost | 229 | 103 | 126 | 44.98% | 44.98% | 44.98% | 5.02 pp | -23 | 18 | -1.28 |
| BTC Market Hours | lstm | LSTM | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 18 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 229 | 112 | 117 | 48.91% | 48.91% | 48.91% | 1.09 pp | -5 | 19 | -0.26 |
| BTC Market Hours Daily | nn | NN | 229 | 109 | 120 | 47.60% | 47.60% | 47.60% | 2.40 pp | -11 | 19 | -0.58 |
| BTC Market Hours Daily | rf | RandomForest | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 19 | -1.84 |
| BTC Market Hours Daily | lstm | LSTM | 229 | 92 | 137 | 40.17% | 40.17% | 40.17% | 9.83 pp | -45 | 19 | -2.37 |

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
