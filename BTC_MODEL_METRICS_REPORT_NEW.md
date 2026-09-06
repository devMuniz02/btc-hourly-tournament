# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T17:24:08.208573+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 246 | 186 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 282 | 222 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 16:00:00+00:00 | 397 | 210 | 187 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 16:00:00+00:00 | 397 | 210 | 187 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 181 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T12:00:00+00:00 | 182 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 17 | 0.47 |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| BTC Market Hours | transformer | Transformer | 210 | 103 | 107 | 49.05% | 49.05% | 49.05% | 0.95 pp | -4 | 17 | -0.24 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| Consolidated Hourly | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 210 | 99 | 111 | 47.14% | 47.14% | 47.14% | 2.86 pp | -12 | 17 | -0.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 212 | 102 | 110 | 48.11% | 48.11% | 48.11% | 1.89 pp | -8 | 10 | -0.80 |
| BTC Market Hours | rf | RandomForest | 210 | 97 | 113 | 46.19% | 46.19% | 46.19% | 3.81 pp | -16 | 17 | -0.94 |
| Consolidated Hourly | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 17 | -1.53 |
| BTC Hourly | transformer | Transformer | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 212 | 96 | 116 | 45.28% | 45.28% | 45.28% | 4.72 pp | -20 | 10 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 17 | -2.12 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Hourly | nn | NN | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 212 | 89 | 123 | 41.98% | 41.98% | 41.98% | 8.02 pp | -34 | 10 | -3.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |
| BTC Hourly | rf | RandomForest | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 8 | -3.75 |
| BTC Daily | rf | RandomForest | 212 | 81 | 131 | 38.21% | 38.21% | 38.21% | 11.79 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 222 | 80 | 142 | 36.04% | 36.04% | 36.04% | 13.96 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 186 | 68 | 118 | 36.56% | 36.56% | 36.56% | 13.44 pp | -50 | 8 | -6.25 |
| BTC Daily | lstm | LSTM | 212 | 71 | 141 | 33.49% | 33.49% | 33.49% | 16.51 pp | -70 | 10 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | transformer | Transformer | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 8 | -1.75 |
| BTC Hourly | nn | NN | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 8 | -3.75 |
| BTC Hourly | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 186 | 68 | 118 | 36.56% | 36.56% | 36.56% | 13.44 pp | -50 | 8 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 212 | 102 | 110 | 48.11% | 48.11% | 48.11% | 1.89 pp | -8 | 10 | -0.80 |
| BTC Daily | nn | NN | 212 | 96 | 116 | 45.28% | 45.28% | 45.28% | 4.72 pp | -20 | 10 | -2.00 |
| BTC Daily | transformer | Transformer | 212 | 89 | 123 | 41.98% | 41.98% | 41.98% | 8.02 pp | -34 | 10 | -3.40 |
| BTC Daily | rf | RandomForest | 212 | 81 | 131 | 38.21% | 38.21% | 38.21% | 11.79 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 222 | 80 | 142 | 36.04% | 36.04% | 36.04% | 13.96 pp | -62 | 11 | -5.64 |
| BTC Daily | lstm | LSTM | 212 | 71 | 141 | 33.49% | 33.49% | 33.49% | 16.51 pp | -70 | 10 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 17 | 0.47 |
| BTC Market Hours | transformer | Transformer | 210 | 103 | 107 | 49.05% | 49.05% | 49.05% | 0.95 pp | -4 | 17 | -0.24 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 210 | 99 | 111 | 47.14% | 47.14% | 47.14% | 2.86 pp | -12 | 17 | -0.71 |
| BTC Market Hours | rf | RandomForest | 210 | 97 | 113 | 46.19% | 46.19% | 46.19% | 3.81 pp | -16 | 17 | -0.94 |
| BTC Market Hours | xgb | XGBoost | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 17 | -1.53 |
| BTC Market Hours | lstm | LSTM | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 17 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 87 | 94 | 48.07% | 48.07% | 48.07% | 1.93 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 83 | 98 | 45.86% | 45.86% | 45.86% | 4.14 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 82 | 99 | 45.30% | 45.30% | 45.30% | 4.70 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 81 | 100 | 44.75% | 44.75% | 44.75% | 5.25 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
