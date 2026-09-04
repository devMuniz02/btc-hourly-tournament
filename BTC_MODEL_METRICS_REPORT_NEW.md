# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T11:05:14.782282+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 209 | 149 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 245 | 185 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 329 | 173 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 329 | 173 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 147 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 148 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 7 | 0.71 |
| BTC Market Hours | nn | NN | 173 | 91 | 82 | 52.60% | 52.60% | 52.60% | 2.60 pp | 9 | 14 | 0.64 |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 15 | -0.20 |
| Consolidated Hourly | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 15 | -0.73 |
| BTC Market Hours | rf | RandomForest | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 14 | -0.79 |
| BTC Market Hours | transformer | Transformer | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 14 | -0.79 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 14 | -0.93 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 14 | -1.79 |
| BTC Market Hours Daily | xgb | XGBoost | 173 | 72 | 101 | 41.62% | 41.62% | 41.62% | 8.38 pp | -29 | 15 | -1.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| BTC Market Hours | lstm | LSTM | 173 | 72 | 101 | 41.62% | 41.62% | 41.62% | 8.38 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 173 | 68 | 105 | 39.31% | 39.31% | 39.31% | 10.69 pp | -37 | 15 | -2.47 |
| BTC Daily | nn | NN | 175 | 77 | 98 | 44.00% | 44.00% | 44.00% | 6.00 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 8 | -2.88 |
| BTC Hourly | nn | NN | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 7 | -3.86 |
| BTC Daily | rf | RandomForest | 175 | 72 | 103 | 41.14% | 41.14% | 41.14% | 8.86 pp | -31 | 8 | -3.88 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 185 | 69 | 116 | 37.30% | 37.30% | 37.30% | 12.70 pp | -47 | 9 | -5.22 |
| BTC Hourly | lstm | LSTM | 149 | 54 | 95 | 36.24% | 36.24% | 36.24% | 13.76 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 149 | 53 | 96 | 35.57% | 35.57% | 35.57% | 14.43 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 175 | 61 | 114 | 34.86% | 34.86% | 34.86% | 15.14 pp | -53 | 8 | -6.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 7 | -3.86 |
| BTC Hourly | lstm | LSTM | 149 | 54 | 95 | 36.24% | 36.24% | 36.24% | 13.76 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 149 | 53 | 96 | 35.57% | 35.57% | 35.57% | 14.43 pp | -43 | 7 | -6.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 8 | -1.62 |
| BTC Daily | nn | NN | 175 | 77 | 98 | 44.00% | 44.00% | 44.00% | 6.00 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 175 | 72 | 103 | 41.14% | 41.14% | 41.14% | 8.86 pp | -31 | 8 | -3.88 |
| BTC Daily | xgb | XGBoost | 185 | 69 | 116 | 37.30% | 37.30% | 37.30% | 12.70 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 175 | 61 | 114 | 34.86% | 34.86% | 34.86% | 15.14 pp | -53 | 8 | -6.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 173 | 91 | 82 | 52.60% | 52.60% | 52.60% | 2.60 pp | 9 | 14 | 0.64 |
| BTC Market Hours | rf | RandomForest | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 14 | -0.79 |
| BTC Market Hours | transformer | Transformer | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 14 | -0.79 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 14 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 173 | 72 | 101 | 41.62% | 41.62% | 41.62% | 8.38 pp | -29 | 14 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 15 | -0.20 |
| BTC Market Hours Daily | nn | NN | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 173 | 72 | 101 | 41.62% | 41.62% | 41.62% | 8.38 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 173 | 68 | 105 | 39.31% | 39.31% | 39.31% | 10.69 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 74 | 73 | 50.34% | 50.34% | 50.34% | 0.34 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 71 | 76 | 48.30% | 48.30% | 48.30% | 1.70 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
