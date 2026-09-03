# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T14:17:37.918419+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 195 | 135 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 231 | 171 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 13:00:00+00:00 | 304 | 159 | 145 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 13:00:00+00:00 | 304 | 159 | 145 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 133 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 134 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 159 | 83 | 76 | 52.20% | 52.20% | 52.20% | 2.20 pp | 7 | 13 | 0.54 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 135 | 69 | 66 | 51.11% | 51.11% | 51.11% | 1.11 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| BTC Hourly | transformer | Transformer | 135 | 67 | 68 | 49.63% | 49.63% | 49.63% | 0.37 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 14 | -0.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 159 | 75 | 84 | 47.17% | 47.17% | 47.17% | 2.83 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 161 | 76 | 85 | 47.20% | 47.20% | 47.20% | 2.80 pp | -9 | 8 | -1.12 |
| BTC Market Hours | rf | RandomForest | 159 | 72 | 87 | 45.28% | 45.28% | 45.28% | 4.72 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 159 | 72 | 87 | 45.28% | 45.28% | 45.28% | 4.72 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | nn | NN | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 14 | -1.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 159 | 68 | 91 | 42.77% | 42.77% | 42.77% | 7.23 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 14 | -2.07 |
| BTC Daily | transformer | Transformer | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 8 | -2.12 |
| BTC Market Hours | lstm | LSTM | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Hourly | nn | NN | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 159 | 61 | 98 | 38.36% | 38.36% | 38.36% | 11.64 pp | -37 | 14 | -2.64 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Daily | rf | RandomForest | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 8 | -2.88 |
| BTC Hourly | rf | RandomForest | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 171 | 64 | 107 | 37.43% | 37.43% | 37.43% | 12.57 pp | -43 | 9 | -4.78 |
| BTC Daily | lstm | LSTM | 161 | 60 | 101 | 37.27% | 37.27% | 37.27% | 12.73 pp | -41 | 8 | -5.12 |
| BTC Hourly | xgb | XGBoost | 135 | 50 | 85 | 37.04% | 37.04% | 37.04% | 12.96 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 135 | 48 | 87 | 35.56% | 35.56% | 35.56% | 14.44 pp | -39 | 6 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 135 | 69 | 66 | 51.11% | 51.11% | 51.11% | 1.11 pp | 3 | 6 | 0.50 |
| BTC Hourly | transformer | Transformer | 135 | 67 | 68 | 49.63% | 49.63% | 49.63% | 0.37 pp | -1 | 6 | -0.17 |
| BTC Hourly | nn | NN | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 6 | -2.50 |
| BTC Hourly | rf | RandomForest | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 135 | 50 | 85 | 37.04% | 37.04% | 37.04% | 12.96 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 135 | 48 | 87 | 35.56% | 35.56% | 35.56% | 14.44 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 8 | -0.38 |
| BTC Daily | nn | NN | 161 | 76 | 85 | 47.20% | 47.20% | 47.20% | 2.80 pp | -9 | 8 | -1.12 |
| BTC Daily | transformer | Transformer | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 8 | -2.12 |
| BTC Daily | rf | RandomForest | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 8 | -2.88 |
| BTC Daily | xgb | XGBoost | 171 | 64 | 107 | 37.43% | 37.43% | 37.43% | 12.57 pp | -43 | 9 | -4.78 |
| BTC Daily | lstm | LSTM | 161 | 60 | 101 | 37.27% | 37.27% | 37.27% | 12.73 pp | -41 | 8 | -5.12 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 159 | 83 | 76 | 52.20% | 52.20% | 52.20% | 2.20 pp | 7 | 13 | 0.54 |
| BTC Market Hours | rf | RandomForest | 159 | 72 | 87 | 45.28% | 45.28% | 45.28% | 4.72 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 159 | 72 | 87 | 45.28% | 45.28% | 45.28% | 4.72 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 13 | -1.31 |
| BTC Market Hours | lstm | LSTM | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 14 | -0.36 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 159 | 75 | 84 | 47.17% | 47.17% | 47.17% | 2.83 pp | -9 | 14 | -0.64 |
| BTC Market Hours Daily | nn | NN | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | rf | RandomForest | 159 | 68 | 91 | 42.77% | 42.77% | 42.77% | 7.23 pp | -23 | 14 | -1.64 |
| BTC Market Hours Daily | xgb | XGBoost | 159 | 65 | 94 | 40.88% | 40.88% | 40.88% | 9.12 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 159 | 61 | 98 | 38.36% | 38.36% | 38.36% | 11.64 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 68 | 65 | 51.13% | 51.13% | 51.13% | 1.13 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 66 | 67 | 49.62% | 49.62% | 49.62% | 0.38 pp | -1 | 11 | -0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
