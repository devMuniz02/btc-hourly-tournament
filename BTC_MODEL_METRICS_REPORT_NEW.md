# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T01:21:51.224716+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 322 | 166 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 322 | 166 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 142 | 75 | 67 | 52.82% | 52.82% | 52.82% | 2.82 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 166 | 86 | 80 | 51.81% | 51.81% | 51.81% | 1.81 pp | 6 | 13 | 0.46 |
| BTC Market Hours Daily | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 166 | 79 | 87 | 47.59% | 47.59% | 47.59% | 2.41 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 166 | 77 | 89 | 46.39% | 46.39% | 46.39% | 3.61 pp | -12 | 13 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 14 | -1.00 |
| BTC Market Hours | transformer | Transformer | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 14 | -1.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 168 | 78 | 90 | 46.43% | 46.43% | 46.43% | 3.57 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| BTC Market Hours Daily | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 14 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 8 | -2.25 |
| BTC Market Hours | lstm | LSTM | 166 | 68 | 98 | 40.96% | 40.96% | 40.96% | 9.04 pp | -30 | 13 | -2.31 |
| BTC Market Hours Daily | lstm | LSTM | 166 | 65 | 101 | 39.16% | 39.16% | 39.16% | 10.84 pp | -36 | 14 | -2.57 |
| BTC Daily | transformer | Transformer | 168 | 73 | 95 | 43.45% | 43.45% | 43.45% | 6.55 pp | -22 | 8 | -2.75 |
| BTC Hourly | nn | NN | 142 | 62 | 80 | 43.66% | 43.66% | 43.66% | 6.34 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 142 | 60 | 82 | 42.25% | 42.25% | 42.25% | 7.75 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 168 | 68 | 100 | 40.48% | 40.48% | 40.48% | 9.52 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 178 | 66 | 112 | 37.08% | 37.08% | 37.08% | 12.92 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 168 | 59 | 109 | 35.12% | 35.12% | 35.12% | 14.88 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 142 | 52 | 90 | 36.62% | 36.62% | 36.62% | 13.38 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 142 | 75 | 67 | 52.82% | 52.82% | 52.82% | 2.82 pp | 8 | 6 | 1.33 |
| BTC Hourly | transformer | Transformer | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 142 | 62 | 80 | 43.66% | 43.66% | 43.66% | 6.34 pp | -18 | 6 | -3.00 |
| BTC Hourly | rf | RandomForest | 142 | 60 | 82 | 42.25% | 42.25% | 42.25% | 7.75 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 142 | 52 | 90 | 36.62% | 36.62% | 36.62% | 13.38 pp | -38 | 6 | -6.33 |
| BTC Hourly | lstm | LSTM | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 6 | -6.67 |

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
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
