# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T00:19:10.005218+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 202 | 142 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 238 | 178 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 23:00:00+00:00 | 321 | 166 | 155 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 23:00:00+00:00 | 321 | 166 | 155 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 139 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 15:00:00+00:00 | 139 | 31 | 108 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 142 | 74 | 68 | 52.11% | 52.11% | 52.11% | 2.11 pp | 6 | 6 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 166 | 86 | 80 | 51.81% | 51.81% | 51.81% | 1.81 pp | 6 | 13 | 0.46 |
| BTC Market Hours Daily | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 14 | -0.29 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 166 | 79 | 87 | 47.59% | 47.59% | 47.59% | 2.41 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 6 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 13 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 14 | -1.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 168 | 78 | 90 | 46.43% | 46.43% | 46.43% | 3.57 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 14 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 13 | -2.00 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 8 | -2.25 |
| BTC Market Hours | lstm | LSTM | 166 | 68 | 98 | 40.96% | 40.96% | 40.96% | 9.04 pp | -30 | 13 | -2.31 |
| BTC Market Hours Daily | lstm | LSTM | 166 | 65 | 101 | 39.16% | 39.16% | 39.16% | 10.84 pp | -36 | 14 | -2.57 |
| BTC Daily | transformer | Transformer | 168 | 73 | 95 | 43.45% | 43.45% | 43.45% | 6.55 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 142 | 61 | 81 | 42.96% | 42.96% | 42.96% | 7.04 pp | -20 | 6 | -3.33 |
| BTC Hourly | rf | RandomForest | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 6 | -4.00 |
| BTC Daily | rf | RandomForest | 168 | 68 | 100 | 40.48% | 40.48% | 40.48% | 9.52 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 178 | 66 | 112 | 37.08% | 37.08% | 37.08% | 12.92 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 168 | 59 | 109 | 35.12% | 35.12% | 35.12% | 14.88 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 142 | 50 | 92 | 35.21% | 35.21% | 35.21% | 14.79 pp | -42 | 6 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 142 | 74 | 68 | 52.11% | 52.11% | 52.11% | 2.11 pp | 6 | 6 | 1.00 |
| BTC Hourly | transformer | Transformer | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 142 | 61 | 81 | 42.96% | 42.96% | 42.96% | 7.04 pp | -20 | 6 | -3.33 |
| BTC Hourly | rf | RandomForest | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 6 | -4.00 |
| BTC Hourly | xgb | XGBoost | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 142 | 50 | 92 | 35.21% | 35.21% | 35.21% | 14.79 pp | -42 | 6 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 168 | 78 | 90 | 46.43% | 46.43% | 46.43% | 3.57 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 168 | 73 | 95 | 43.45% | 43.45% | 43.45% | 6.55 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 168 | 68 | 100 | 40.48% | 40.48% | 40.48% | 9.52 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 178 | 66 | 112 | 37.08% | 37.08% | 37.08% | 12.92 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 168 | 59 | 109 | 35.12% | 35.12% | 35.12% | 14.88 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 166 | 86 | 80 | 51.81% | 51.81% | 51.81% | 1.81 pp | 6 | 13 | 0.46 |
| BTC Market Hours | rf | RandomForest | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 13 | -0.92 |
| BTC Market Hours | transformer | Transformer | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 166 | 68 | 98 | 40.96% | 40.96% | 40.96% | 9.04 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 166 | 79 | 87 | 47.59% | 47.59% | 47.59% | 2.41 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 14 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 14 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 166 | 65 | 101 | 39.16% | 39.16% | 39.16% | 10.84 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 139 | 73 | 66 | 52.52% | 52.52% | 52.52% | 2.52 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 139 | 66 | 73 | 47.48% | 47.48% | 47.48% | 2.52 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 139 | 65 | 74 | 46.76% | 46.76% | 46.76% | 3.24 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 139 | 64 | 75 | 46.04% | 46.04% | 46.04% | 3.96 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 139 | 58 | 81 | 41.73% | 41.73% | 41.73% | 8.27 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
