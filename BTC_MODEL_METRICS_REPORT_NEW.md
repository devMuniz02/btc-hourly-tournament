# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T14:16:11.091255+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 211 | 151 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 247 | 187 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 333 | 175 | 158 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 333 | 175 | 158 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 149 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T20:00:00+00:00 | 150 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 151 | 79 | 72 | 52.32% | 52.32% | 52.32% | 2.32 pp | 7 | 7 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 175 | 92 | 83 | 52.57% | 52.57% | 52.57% | 2.57 pp | 9 | 14 | 0.64 |
| Consolidated Hourly | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| BTC Market Hours Daily | transformer | Transformer | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 15 | 0.07 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Market Hours | transformer | Transformer | 175 | 83 | 92 | 47.43% | 47.43% | 47.43% | 2.57 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| Consolidated Hourly | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 175 | 75 | 100 | 42.86% | 42.86% | 42.86% | 7.14 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | xgb | XGBoost | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 15 | -1.93 |
| BTC Market Hours | lstm | LSTM | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 175 | 70 | 105 | 40.00% | 40.00% | 40.00% | 10.00 pp | -35 | 15 | -2.33 |
| BTC Daily | nn | NN | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 8 | -3.12 |
| BTC Hourly | nn | NN | 151 | 64 | 87 | 42.38% | 42.38% | 42.38% | 7.62 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 177 | 72 | 105 | 40.68% | 40.68% | 40.68% | 9.32 pp | -33 | 8 | -4.12 |
| BTC Hourly | rf | RandomForest | 151 | 61 | 90 | 40.40% | 40.40% | 40.40% | 9.60 pp | -29 | 7 | -4.14 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 9 | -5.44 |
| BTC Hourly | lstm | LSTM | 151 | 54 | 97 | 35.76% | 35.76% | 35.76% | 14.24 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 151 | 53 | 98 | 35.10% | 35.10% | 35.10% | 14.90 pp | -45 | 7 | -6.43 |
| BTC Daily | lstm | LSTM | 177 | 62 | 115 | 35.03% | 35.03% | 35.03% | 14.97 pp | -53 | 8 | -6.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 151 | 79 | 72 | 52.32% | 52.32% | 52.32% | 2.32 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 151 | 64 | 87 | 42.38% | 42.38% | 42.38% | 7.62 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 151 | 61 | 90 | 40.40% | 40.40% | 40.40% | 9.60 pp | -29 | 7 | -4.14 |
| BTC Hourly | lstm | LSTM | 151 | 54 | 97 | 35.76% | 35.76% | 35.76% | 14.24 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 151 | 53 | 98 | 35.10% | 35.10% | 35.10% | 14.90 pp | -45 | 7 | -6.43 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| BTC Daily | nn | NN | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 8 | -3.12 |
| BTC Daily | rf | RandomForest | 177 | 72 | 105 | 40.68% | 40.68% | 40.68% | 9.32 pp | -33 | 8 | -4.12 |
| BTC Daily | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 9 | -5.44 |
| BTC Daily | lstm | LSTM | 177 | 62 | 115 | 35.03% | 35.03% | 35.03% | 14.97 pp | -53 | 8 | -6.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 175 | 92 | 83 | 52.57% | 52.57% | 52.57% | 2.57 pp | 9 | 14 | 0.64 |
| BTC Market Hours | transformer | Transformer | 175 | 83 | 92 | 47.43% | 47.43% | 47.43% | 2.57 pp | -9 | 14 | -0.64 |
| BTC Market Hours | rf | RandomForest | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 14 | -0.93 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 14 | -1.07 |
| BTC Market Hours | xgb | XGBoost | 175 | 75 | 100 | 42.86% | 42.86% | 42.86% | 7.14 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 14 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 15 | 0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 15 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 175 | 70 | 105 | 40.00% | 40.00% | 40.00% | 10.00 pp | -35 | 15 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 75 | 74 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 65 | 84 | 43.62% | 43.62% | 43.62% | 6.38 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
