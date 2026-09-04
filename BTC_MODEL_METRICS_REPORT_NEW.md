# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T03:12:58.163177+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 204 | 144 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 240 | 180 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 324 | 168 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 323 | 167 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 144 | 75 | 69 | 52.08% | 52.08% | 52.08% | 2.08 pp | 6 | 6 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 168 | 87 | 81 | 51.79% | 51.79% | 51.79% | 1.79 pp | 6 | 13 | 0.46 |
| BTC Market Hours Daily | transformer | Transformer | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 167 | 80 | 87 | 47.90% | 47.90% | 47.90% | 2.10 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 168 | 79 | 89 | 47.02% | 47.02% | 47.02% | 2.98 pp | -10 | 13 | -0.77 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | nn | NN | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Market Hours | transformer | Transformer | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| BTC Daily | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 8 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | xgb | XGBoost | 167 | 70 | 97 | 41.92% | 41.92% | 41.92% | 8.08 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 8 | -2.25 |
| BTC Market Hours | lstm | LSTM | 168 | 69 | 99 | 41.07% | 41.07% | 41.07% | 8.93 pp | -30 | 13 | -2.31 |
| BTC Daily | transformer | Transformer | 170 | 75 | 95 | 44.12% | 44.12% | 44.12% | 5.88 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 167 | 65 | 102 | 38.92% | 38.92% | 38.92% | 11.08 pp | -37 | 14 | -2.64 |
| BTC Hourly | nn | NN | 144 | 62 | 82 | 43.06% | 43.06% | 43.06% | 6.94 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | rf | RandomForest | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 8 | -3.75 |
| BTC Hourly | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 6 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 170 | 60 | 110 | 35.29% | 35.29% | 35.29% | 14.71 pp | -50 | 8 | -6.25 |
| BTC Hourly | xgb | XGBoost | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 6 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 144 | 75 | 69 | 52.08% | 52.08% | 52.08% | 2.08 pp | 6 | 6 | 1.00 |
| BTC Hourly | transformer | Transformer | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 144 | 62 | 82 | 43.06% | 43.06% | 43.06% | 6.94 pp | -20 | 6 | -3.33 |
| BTC Hourly | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 6 | -4.00 |
| BTC Hourly | xgb | XGBoost | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 6 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 170 | 78 | 92 | 45.88% | 45.88% | 45.88% | 4.12 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 170 | 76 | 94 | 44.71% | 44.71% | 44.71% | 5.29 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 170 | 75 | 95 | 44.12% | 44.12% | 44.12% | 5.88 pp | -20 | 8 | -2.50 |
| BTC Daily | rf | RandomForest | 170 | 70 | 100 | 41.18% | 41.18% | 41.18% | 8.82 pp | -30 | 8 | -3.75 |
| BTC Daily | xgb | XGBoost | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 9 | -5.11 |
| BTC Daily | lstm | LSTM | 170 | 60 | 110 | 35.29% | 35.29% | 35.29% | 14.71 pp | -50 | 8 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 168 | 87 | 81 | 51.79% | 51.79% | 51.79% | 1.79 pp | 6 | 13 | 0.46 |
| BTC Market Hours | rf | RandomForest | 168 | 79 | 89 | 47.02% | 47.02% | 47.02% | 2.98 pp | -10 | 13 | -0.77 |
| BTC Market Hours | transformer | Transformer | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 13 | -1.85 |
| BTC Market Hours | lstm | LSTM | 168 | 69 | 99 | 41.07% | 41.07% | 41.07% | 8.93 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 167 | 80 | 87 | 47.90% | 47.90% | 47.90% | 2.10 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 14 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 167 | 70 | 97 | 41.92% | 41.92% | 41.92% | 8.08 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 167 | 65 | 102 | 38.92% | 38.92% | 38.92% | 11.08 pp | -37 | 14 | -2.64 |

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
