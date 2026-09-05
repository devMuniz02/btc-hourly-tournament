# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T11:17:22.894397+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 226 | 166 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 262 | 202 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 359 | 190 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 358 | 189 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 161 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 161 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 43 | 118 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 43 | 118 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 189 | 99 | 90 | 52.38% | 52.38% | 52.38% | 2.38 pp | 9 | 16 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 189 | 91 | 98 | 48.15% | 48.15% | 48.15% | 1.85 pp | -7 | 16 | -0.44 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 16 | -0.81 |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 16 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| Consolidated Hourly | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 189 | 78 | 111 | 41.27% | 41.27% | 41.27% | 8.73 pp | -33 | 16 | -2.06 |
| Consolidated Hourly | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 189 | 76 | 113 | 40.21% | 40.21% | 40.21% | 9.79 pp | -37 | 16 | -2.31 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 189 | 99 | 90 | 52.38% | 52.38% | 52.38% | 2.38 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 189 | 91 | 98 | 48.15% | 48.15% | 48.15% | 1.85 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 16 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 189 | 78 | 111 | 41.27% | 41.27% | 41.27% | 8.73 pp | -33 | 16 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 189 | 76 | 113 | 40.21% | 40.21% | 40.21% | 9.79 pp | -37 | 16 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
