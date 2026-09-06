# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T03:17:41.862638+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 237 | 177 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 272 | 212 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 382 | 200 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 382 | 200 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 173 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 173 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 49 | 124 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 49 | 124 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 200 | 103 | 97 | 51.50% | 51.50% | 51.50% | 1.50 pp | 6 | 17 | 0.35 |
| BTC Market Hours | nn | NN | 200 | 102 | 98 | 51.00% | 51.00% | 51.00% | 1.00 pp | 4 | 16 | 0.25 |
| Consolidated Hourly | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 16 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 200 | 97 | 103 | 48.50% | 48.50% | 48.50% | 1.50 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 200 | 95 | 105 | 47.50% | 47.50% | 47.50% | 2.50 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 200 | 94 | 106 | 47.00% | 47.00% | 47.00% | 3.00 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| BTC Daily | mlp_sklearn | MLPClassifier | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 9 | -1.11 |
| BTC Market Hours | rf | RandomForest | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 16 | -1.12 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 17 | -1.53 |
| BTC Hourly | transformer | Transformer | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 200 | 86 | 114 | 43.00% | 43.00% | 43.00% | 7.00 pp | -28 | 16 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 16 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 200 | 82 | 118 | 41.00% | 41.00% | 41.00% | 9.00 pp | -36 | 17 | -2.12 |
| BTC Daily | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 200 | 81 | 119 | 40.50% | 40.50% | 40.50% | 9.50 pp | -38 | 17 | -2.24 |
| Consolidated Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Hourly | rf | RandomForest | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 202 | 85 | 117 | 42.08% | 42.08% | 42.08% | 7.92 pp | -32 | 9 | -3.56 |
| BTC Hourly | nn | NN | 177 | 74 | 103 | 41.81% | 41.81% | 41.81% | 8.19 pp | -29 | 8 | -3.62 |
| BTC Daily | rf | RandomForest | 202 | 76 | 126 | 37.62% | 37.62% | 37.62% | 12.38 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 177 | 64 | 113 | 36.16% | 36.16% | 36.16% | 13.84 pp | -49 | 8 | -6.12 |
| BTC Daily | xgb | XGBoost | 212 | 75 | 137 | 35.38% | 35.38% | 35.38% | 14.62 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 202 | 68 | 134 | 33.66% | 33.66% | 33.66% | 16.34 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| BTC Hourly | rf | RandomForest | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 177 | 74 | 103 | 41.81% | 41.81% | 41.81% | 8.19 pp | -29 | 8 | -3.62 |
| BTC Hourly | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 177 | 64 | 113 | 36.16% | 36.16% | 36.16% | 13.84 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 9 | -1.11 |
| BTC Daily | nn | NN | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 202 | 85 | 117 | 42.08% | 42.08% | 42.08% | 7.92 pp | -32 | 9 | -3.56 |
| BTC Daily | rf | RandomForest | 202 | 76 | 126 | 37.62% | 37.62% | 37.62% | 12.38 pp | -50 | 9 | -5.56 |
| BTC Daily | xgb | XGBoost | 212 | 75 | 137 | 35.38% | 35.38% | 35.38% | 14.62 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 202 | 68 | 134 | 33.66% | 33.66% | 33.66% | 16.34 pp | -66 | 9 | -7.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 200 | 102 | 98 | 51.00% | 51.00% | 51.00% | 1.00 pp | 4 | 16 | 0.25 |
| BTC Market Hours | transformer | Transformer | 200 | 98 | 102 | 49.00% | 49.00% | 49.00% | 1.00 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 200 | 94 | 106 | 47.00% | 47.00% | 47.00% | 3.00 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 200 | 86 | 114 | 43.00% | 43.00% | 43.00% | 7.00 pp | -28 | 16 | -1.75 |
| BTC Market Hours | lstm | LSTM | 200 | 84 | 116 | 42.00% | 42.00% | 42.00% | 8.00 pp | -32 | 16 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 200 | 103 | 97 | 51.50% | 51.50% | 51.50% | 1.50 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 200 | 97 | 103 | 48.50% | 48.50% | 48.50% | 1.50 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 200 | 95 | 105 | 47.50% | 47.50% | 47.50% | 2.50 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 200 | 87 | 113 | 43.50% | 43.50% | 43.50% | 6.50 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 200 | 82 | 118 | 41.00% | 41.00% | 41.00% | 9.00 pp | -36 | 17 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 200 | 81 | 119 | 40.50% | 40.50% | 40.50% | 9.50 pp | -38 | 17 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
