# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T12:38:42.523654+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 178 | 118 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 214 | 154 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 272 | 142 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 272 | 142 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 119 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 142 | 73 | 69 | 51.41% | 51.41% | 51.41% | 1.41 pp | 4 | 11 | 0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | transformer | Transformer | 142 | 67 | 75 | 47.18% | 47.18% | 47.18% | 2.82 pp | -8 | 12 | -0.67 |
| BTC Hourly | transformer | Transformer | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 5 | -0.80 |
| BTC Market Hours | rf | RandomForest | 142 | 66 | 76 | 46.48% | 46.48% | 46.48% | 3.52 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 12 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 11 | -2.18 |
| BTC Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 142 | 57 | 85 | 40.14% | 40.14% | 40.14% | 9.86 pp | -28 | 11 | -2.55 |
| BTC Daily | transformer | Transformer | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 142 | 54 | 88 | 38.03% | 38.03% | 38.03% | 11.97 pp | -34 | 12 | -2.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| BTC Hourly | rf | RandomForest | 118 | 49 | 69 | 41.53% | 41.53% | 41.53% | 8.47 pp | -20 | 5 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| BTC Daily | xgb | XGBoost | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 8 | -5.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| BTC Daily | lstm | LSTM | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 7 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 118 | 43 | 75 | 36.44% | 36.44% | 36.44% | 13.56 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 118 | 39 | 79 | 33.05% | 33.05% | 33.05% | 16.95 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | transformer | Transformer | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 5 | -0.80 |
| BTC Hourly | nn | NN | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 118 | 49 | 69 | 41.53% | 41.53% | 41.53% | 8.47 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 118 | 43 | 75 | 36.44% | 36.44% | 36.44% | 13.56 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 118 | 39 | 79 | 33.05% | 33.05% | 33.05% | 16.95 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 144 | 63 | 81 | 43.75% | 43.75% | 43.75% | 6.25 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 7 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 142 | 73 | 69 | 51.41% | 51.41% | 51.41% | 1.41 pp | 4 | 11 | 0.36 |
| BTC Market Hours | rf | RandomForest | 142 | 66 | 76 | 46.48% | 46.48% | 46.48% | 3.52 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | xgb | XGBoost | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 142 | 57 | 85 | 40.14% | 40.14% | 40.14% | 9.86 pp | -28 | 11 | -2.55 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 142 | 67 | 75 | 47.18% | 47.18% | 47.18% | 2.82 pp | -8 | 12 | -0.67 |
| BTC Market Hours Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 142 | 54 | 88 | 38.03% | 38.03% | 38.03% | 11.97 pp | -34 | 12 | -2.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 118 | 59 | 59 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 118 | 51 | 67 | 43.22% | 43.22% | 43.22% | 6.78 pp | -16 | 10 | -1.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
