# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T12:12:50.720532+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 162 | 102 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 198 | 138 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 243 | 126 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 243 | 126 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 103 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 103 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 11 | 92 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 21:00:00+00:00 | 103 | 11 | 92 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 126 | 67 | 59 | 53.17% | 53.17% | 53.17% | 3.17 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 126 | 61 | 65 | 48.41% | 48.41% | 48.41% | 1.59 pp | -4 | 11 | -0.36 |
| BTC Market Hours | rf | RandomForest | 126 | 61 | 65 | 48.41% | 48.41% | 48.41% | 1.59 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 102 | 49 | 53 | 48.04% | 48.04% | 48.04% | 1.96 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 11 | -1.09 |
| BTC Hourly | nn | NN | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 126 | 56 | 70 | 44.44% | 44.44% | 44.44% | 5.56 pp | -14 | 11 | -1.27 |
| BTC Daily | nn | NN | 128 | 60 | 68 | 46.88% | 46.88% | 46.88% | 3.12 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| BTC Daily | transformer | Transformer | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 126 | 51 | 75 | 40.48% | 40.48% | 40.48% | 9.52 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 126 | 49 | 77 | 38.89% | 38.89% | 38.89% | 11.11 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| BTC Market Hours | lstm | LSTM | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 10 | -3.20 |
| BTC Hourly | rf | RandomForest | 102 | 42 | 60 | 41.18% | 41.18% | 41.18% | 8.82 pp | -18 | 5 | -3.60 |
| BTC Daily | rf | RandomForest | 128 | 53 | 75 | 41.41% | 41.41% | 41.41% | 8.59 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 138 | 50 | 88 | 36.23% | 36.23% | 36.23% | 13.77 pp | -38 | 7 | -5.43 |
| BTC Hourly | xgb | XGBoost | 102 | 35 | 67 | 34.31% | 34.31% | 34.31% | 15.69 pp | -32 | 5 | -6.40 |
| BTC Daily | lstm | LSTM | 128 | 44 | 84 | 34.38% | 34.38% | 34.38% | 15.62 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 102 | 31 | 71 | 30.39% | 30.39% | 30.39% | 19.61 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 102 | 49 | 53 | 48.04% | 48.04% | 48.04% | 1.96 pp | -4 | 5 | -0.80 |
| BTC Hourly | nn | NN | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 102 | 42 | 60 | 41.18% | 41.18% | 41.18% | 8.82 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 102 | 35 | 67 | 34.31% | 34.31% | 34.31% | 15.69 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 102 | 31 | 71 | 30.39% | 30.39% | 30.39% | 19.61 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | nn | NN | 128 | 60 | 68 | 46.88% | 46.88% | 46.88% | 3.12 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 128 | 53 | 75 | 41.41% | 41.41% | 41.41% | 8.59 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 138 | 50 | 88 | 36.23% | 36.23% | 36.23% | 13.77 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 128 | 44 | 84 | 34.38% | 34.38% | 34.38% | 15.62 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 126 | 67 | 59 | 53.17% | 53.17% | 53.17% | 3.17 pp | 8 | 10 | 0.80 |
| BTC Market Hours | rf | RandomForest | 126 | 61 | 65 | 48.41% | 48.41% | 48.41% | 1.59 pp | -4 | 10 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 10 | -1.20 |
| BTC Market Hours | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 126 | 51 | 75 | 40.48% | 40.48% | 40.48% | 9.52 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 10 | -3.20 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 126 | 61 | 65 | 48.41% | 48.41% | 48.41% | 1.59 pp | -4 | 11 | -0.36 |
| BTC Market Hours Daily | transformer | Transformer | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | rf | RandomForest | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | nn | NN | 126 | 56 | 70 | 44.44% | 44.44% | 44.44% | 5.56 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 126 | 49 | 77 | 38.89% | 38.89% | 38.89% | 11.11 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 11 | -2.91 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 103 | 55 | 48 | 53.40% | 53.40% | 53.40% | 3.40 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 103 | 53 | 50 | 51.46% | 51.46% | 51.46% | 1.46 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 103 | 48 | 55 | 46.60% | 46.60% | 46.60% | 3.40 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 4 | 7 | 36.36% | 36.36% | 36.36% | 13.64 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
