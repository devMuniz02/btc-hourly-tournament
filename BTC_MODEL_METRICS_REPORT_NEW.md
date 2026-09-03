# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T08:07:09.711538+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 191 | 131 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 227 | 167 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 298 | 155 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 298 | 155 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 131 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | transformer | Transformer | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 7 | 0.14 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 13 | -0.69 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | nn | NN | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 13 | -1.15 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Daily | nn | NN | 157 | 74 | 83 | 47.13% | 47.13% | 47.13% | 2.87 pp | -9 | 7 | -1.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 7 | -2.14 |
| BTC Hourly | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 6 | -2.17 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| BTC Daily | rf | RandomForest | 157 | 68 | 89 | 43.31% | 43.31% | 43.31% | 6.69 pp | -21 | 7 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 13 | -3.15 |
| BTC Hourly | rf | RandomForest | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 131 | 49 | 82 | 37.40% | 37.40% | 37.40% | 12.60 pp | -33 | 6 | -5.50 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| BTC Daily | lstm | LSTM | 157 | 59 | 98 | 37.58% | 37.58% | 37.58% | 12.42 pp | -39 | 7 | -5.57 |
| BTC Daily | xgb | XGBoost | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 8 | -5.62 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| BTC Hourly | lstm | LSTM | 131 | 47 | 84 | 35.88% | 35.88% | 35.88% | 14.12 pp | -37 | 6 | -6.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | transformer | Transformer | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 6 | -2.17 |
| BTC Hourly | rf | RandomForest | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 131 | 49 | 82 | 37.40% | 37.40% | 37.40% | 12.60 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 131 | 47 | 84 | 35.88% | 35.88% | 35.88% | 14.12 pp | -37 | 6 | -6.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 79 | 78 | 50.32% | 50.32% | 50.32% | 0.32 pp | 1 | 7 | 0.14 |
| BTC Daily | nn | NN | 157 | 74 | 83 | 47.13% | 47.13% | 47.13% | 2.87 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 7 | -2.14 |
| BTC Daily | rf | RandomForest | 157 | 68 | 89 | 43.31% | 43.31% | 43.31% | 6.69 pp | -21 | 7 | -3.00 |
| BTC Daily | lstm | LSTM | 157 | 59 | 98 | 37.58% | 37.58% | 37.58% | 12.42 pp | -39 | 7 | -5.57 |
| BTC Daily | xgb | XGBoost | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 8 | -5.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 13 | -0.69 |
| BTC Market Hours Daily | nn | NN | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 13 | -3.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
