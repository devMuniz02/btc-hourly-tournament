# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T11:09:28.337408+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 274 | 214 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 310 | 250 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 446 | 238 | 208 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 00:00:00+00:00 | 446 | 238 | 208 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 207 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T13:00:00+00:00 | 208 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 238 | 121 | 117 | 50.84% | 50.84% | 50.84% | 0.84 pp | 4 | 19 | 0.21 |
| BTC Market Hours Daily | transformer | Transformer | 238 | 117 | 121 | 49.16% | 49.16% | 49.16% | 0.84 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 238 | 116 | 122 | 48.74% | 48.74% | 48.74% | 1.26 pp | -6 | 20 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 214 | 105 | 109 | 49.07% | 49.07% | 49.07% | 0.93 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | nn | NN | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 20 | -0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 19 | -0.74 |
| BTC Market Hours | transformer | Transformer | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 19 | -0.74 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours | xgb | XGBoost | 238 | 110 | 128 | 46.22% | 46.22% | 46.22% | 3.78 pp | -18 | 19 | -0.95 |
| Consolidated Hourly | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 19 | -1.16 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 238 | 105 | 133 | 44.12% | 44.12% | 44.12% | 5.88 pp | -28 | 20 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 11 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| BTC Daily | nn | NN | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 11 | -1.82 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 238 | 101 | 137 | 42.44% | 42.44% | 42.44% | 7.56 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 238 | 97 | 141 | 40.76% | 40.76% | 40.76% | 9.24 pp | -44 | 20 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| BTC Hourly | transformer | Transformer | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 214 | 90 | 124 | 42.06% | 42.06% | 42.06% | 7.94 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 240 | 98 | 142 | 40.83% | 40.83% | 40.83% | 9.17 pp | -44 | 11 | -4.00 |
| BTC Hourly | rf | RandomForest | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 9 | -4.22 |
| BTC Daily | rf | RandomForest | 240 | 92 | 148 | 38.33% | 38.33% | 38.33% | 11.67 pp | -56 | 11 | -5.09 |
| BTC Daily | xgb | XGBoost | 250 | 90 | 160 | 36.00% | 36.25% | 36.00% | 14.00 pp | -70 | 12 | -5.83 |
| BTC Hourly | lstm | LSTM | 214 | 79 | 135 | 36.92% | 36.92% | 36.92% | 13.08 pp | -56 | 9 | -6.22 |
| BTC Daily | lstm | LSTM | 240 | 81 | 159 | 33.75% | 33.75% | 33.75% | 16.25 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 214 | 73 | 141 | 34.11% | 34.11% | 34.11% | 15.89 pp | -68 | 9 | -7.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 214 | 105 | 109 | 49.07% | 49.07% | 49.07% | 0.93 pp | -4 | 9 | -0.44 |
| BTC Hourly | transformer | Transformer | 214 | 96 | 118 | 44.86% | 44.86% | 44.86% | 5.14 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 214 | 90 | 124 | 42.06% | 42.06% | 42.06% | 7.94 pp | -34 | 9 | -3.78 |
| BTC Hourly | rf | RandomForest | 214 | 88 | 126 | 41.12% | 41.12% | 41.12% | 8.88 pp | -38 | 9 | -4.22 |
| BTC Hourly | lstm | LSTM | 214 | 79 | 135 | 36.92% | 36.92% | 36.92% | 13.08 pp | -56 | 9 | -6.22 |
| BTC Hourly | xgb | XGBoost | 214 | 73 | 141 | 34.11% | 34.11% | 34.11% | 15.89 pp | -68 | 9 | -7.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 240 | 112 | 128 | 46.67% | 46.67% | 46.67% | 3.33 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 240 | 110 | 130 | 45.83% | 45.83% | 45.83% | 4.17 pp | -20 | 11 | -1.82 |
| BTC Daily | transformer | Transformer | 240 | 98 | 142 | 40.83% | 40.83% | 40.83% | 9.17 pp | -44 | 11 | -4.00 |
| BTC Daily | rf | RandomForest | 240 | 92 | 148 | 38.33% | 38.33% | 38.33% | 11.67 pp | -56 | 11 | -5.09 |
| BTC Daily | xgb | XGBoost | 250 | 90 | 160 | 36.00% | 36.25% | 36.00% | 14.00 pp | -70 | 12 | -5.83 |
| BTC Daily | lstm | LSTM | 240 | 81 | 159 | 33.75% | 33.75% | 33.75% | 16.25 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 238 | 121 | 117 | 50.84% | 50.84% | 50.84% | 0.84 pp | 4 | 19 | 0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 19 | -0.74 |
| BTC Market Hours | transformer | Transformer | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 19 | -0.74 |
| BTC Market Hours | xgb | XGBoost | 238 | 110 | 128 | 46.22% | 46.22% | 46.22% | 3.78 pp | -18 | 19 | -0.95 |
| BTC Market Hours | rf | RandomForest | 238 | 108 | 130 | 45.38% | 45.38% | 45.38% | 4.62 pp | -22 | 19 | -1.16 |
| BTC Market Hours | lstm | LSTM | 238 | 101 | 137 | 42.44% | 42.44% | 42.44% | 7.56 pp | -36 | 19 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 238 | 117 | 121 | 49.16% | 49.16% | 49.16% | 0.84 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 238 | 116 | 122 | 48.74% | 48.74% | 48.74% | 1.26 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 238 | 112 | 126 | 47.06% | 47.06% | 47.06% | 2.94 pp | -14 | 20 | -0.70 |
| BTC Market Hours Daily | rf | RandomForest | 238 | 105 | 133 | 44.12% | 44.12% | 44.12% | 5.88 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 238 | 104 | 134 | 43.70% | 43.70% | 43.70% | 6.30 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 238 | 97 | 141 | 40.76% | 40.76% | 40.76% | 9.24 pp | -44 | 20 | -2.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 207 | 101 | 106 | 48.79% | 48.79% | 48.79% | 1.21 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 207 | 99 | 108 | 47.83% | 47.83% | 47.83% | 2.17 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 207 | 96 | 111 | 46.38% | 46.38% | 46.38% | 3.62 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 207 | 94 | 113 | 45.41% | 45.41% | 45.41% | 4.59 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 207 | 92 | 115 | 44.44% | 44.44% | 44.44% | 5.56 pp | -23 | 14 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 207 | 89 | 118 | 43.00% | 43.00% | 43.00% | 7.00 pp | -29 | 14 | -2.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 68 | 32 | 36 | 47.06% | 47.06% | 47.06% | 2.94 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 68 | 29 | 39 | 42.65% | 42.65% | 42.65% | 7.35 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 68 | 27 | 41 | 39.71% | 39.71% | 39.71% | 10.29 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | nn | NN | 68 | 26 | 42 | 38.24% | 38.24% | 38.24% | 11.76 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
