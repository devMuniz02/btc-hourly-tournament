# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T21:34:45.945094+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 249 | 189 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 285 | 225 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 404 | 213 | 191 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 20:00:00+00:00 | 404 | 213 | 191 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 184 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 213 | 110 | 103 | 51.64% | 51.64% | 51.64% | 1.64 pp | 7 | 18 | 0.39 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 189 | 96 | 93 | 50.79% | 50.79% | 50.79% | 0.79 pp | 3 | 8 | 0.38 |
| BTC Market Hours | nn | NN | 213 | 109 | 104 | 51.17% | 51.17% | 51.17% | 1.17 pp | 5 | 17 | 0.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 18 | -0.28 |
| BTC Market Hours | transformer | Transformer | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 17 | -0.29 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | nn | NN | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 213 | 100 | 113 | 46.95% | 46.95% | 46.95% | 3.05 pp | -13 | 17 | -0.76 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| BTC Market Hours | rf | RandomForest | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 17 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 213 | 95 | 118 | 44.60% | 44.60% | 44.60% | 5.40 pp | -23 | 18 | -1.28 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 17 | -1.59 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Hourly | transformer | Transformer | 189 | 87 | 102 | 46.03% | 46.03% | 46.03% | 3.97 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Daily | nn | NN | 215 | 97 | 118 | 45.12% | 45.12% | 45.12% | 4.88 pp | -21 | 10 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 213 | 87 | 126 | 40.85% | 40.85% | 40.85% | 9.15 pp | -39 | 18 | -2.17 |
| BTC Market Hours | lstm | LSTM | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 17 | -2.18 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 213 | 85 | 128 | 39.91% | 39.91% | 39.91% | 10.09 pp | -43 | 18 | -2.39 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 8 | -3.12 |
| BTC Daily | transformer | Transformer | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 189 | 79 | 110 | 41.80% | 41.80% | 41.80% | 8.20 pp | -31 | 8 | -3.88 |
| BTC Daily | rf | RandomForest | 215 | 83 | 132 | 38.60% | 38.60% | 38.60% | 11.40 pp | -49 | 10 | -4.90 |
| BTC Daily | xgb | XGBoost | 225 | 81 | 144 | 36.00% | 36.00% | 36.00% | 14.00 pp | -63 | 11 | -5.73 |
| BTC Hourly | lstm | LSTM | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 215 | 73 | 142 | 33.95% | 33.95% | 33.95% | 16.05 pp | -69 | 10 | -6.90 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 189 | 96 | 93 | 50.79% | 50.79% | 50.79% | 0.79 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 189 | 87 | 102 | 46.03% | 46.03% | 46.03% | 3.97 pp | -15 | 8 | -1.88 |
| BTC Hourly | nn | NN | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 189 | 79 | 110 | 41.80% | 41.80% | 41.80% | 8.20 pp | -31 | 8 | -3.88 |
| BTC Hourly | lstm | LSTM | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 189 | 70 | 119 | 37.04% | 37.04% | 37.04% | 12.96 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 215 | 102 | 113 | 47.44% | 47.44% | 47.44% | 2.56 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 215 | 97 | 118 | 45.12% | 45.12% | 45.12% | 4.88 pp | -21 | 10 | -2.10 |
| BTC Daily | transformer | Transformer | 215 | 89 | 126 | 41.40% | 41.40% | 41.40% | 8.60 pp | -37 | 10 | -3.70 |
| BTC Daily | rf | RandomForest | 215 | 83 | 132 | 38.60% | 38.60% | 38.60% | 11.40 pp | -49 | 10 | -4.90 |
| BTC Daily | xgb | XGBoost | 225 | 81 | 144 | 36.00% | 36.00% | 36.00% | 14.00 pp | -63 | 11 | -5.73 |
| BTC Daily | lstm | LSTM | 215 | 73 | 142 | 33.95% | 33.95% | 33.95% | 16.05 pp | -69 | 10 | -6.90 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 213 | 109 | 104 | 51.17% | 51.17% | 51.17% | 1.17 pp | 5 | 17 | 0.29 |
| BTC Market Hours | transformer | Transformer | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 17 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 213 | 100 | 113 | 46.95% | 46.95% | 46.95% | 3.05 pp | -13 | 17 | -0.76 |
| BTC Market Hours | rf | RandomForest | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 17 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 17 | -1.59 |
| BTC Market Hours | lstm | LSTM | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 17 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 213 | 110 | 103 | 51.64% | 51.64% | 51.64% | 1.64 pp | 7 | 18 | 0.39 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | nn | NN | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 213 | 95 | 118 | 44.60% | 44.60% | 44.60% | 5.40 pp | -23 | 18 | -1.28 |
| BTC Market Hours Daily | xgb | XGBoost | 213 | 87 | 126 | 40.85% | 40.85% | 40.85% | 9.15 pp | -39 | 18 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 213 | 85 | 128 | 39.91% | 39.91% | 39.91% | 10.09 pp | -43 | 18 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
