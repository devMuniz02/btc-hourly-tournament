# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T20:44:05.367553+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 183 | 123 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 219 | 159 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 19:00:00+00:00 | 285 | 147 | 138 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 19:00:00+00:00 | 285 | 147 | 138 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 124 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 147 | 77 | 70 | 52.38% | 52.38% | 52.38% | 2.38 pp | 7 | 12 | 0.58 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 123 | 61 | 62 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 6 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 149 | 73 | 76 | 48.99% | 48.99% | 48.99% | 1.01 pp | -3 | 7 | -0.43 |
| BTC Hourly | transformer | Transformer | 123 | 60 | 63 | 48.78% | 48.78% | 48.78% | 1.22 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 13 | -0.54 |
| BTC Market Hours | rf | RandomForest | 147 | 69 | 78 | 46.94% | 46.94% | 46.94% | 3.06 pp | -9 | 12 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 13 | -1.31 |
| BTC Market Hours | transformer | Transformer | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 12 | -1.42 |
| BTC Hourly | nn | NN | 123 | 57 | 66 | 46.34% | 46.34% | 46.34% | 3.66 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |
| BTC Market Hours | xgb | XGBoost | 147 | 62 | 85 | 42.18% | 42.18% | 42.18% | 7.82 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 147 | 60 | 87 | 40.82% | 40.82% | 40.82% | 9.18 pp | -27 | 13 | -2.08 |
| BTC Daily | nn | NN | 149 | 67 | 82 | 44.97% | 44.97% | 44.97% | 5.03 pp | -15 | 7 | -2.14 |
| BTC Market Hours | lstm | LSTM | 147 | 59 | 88 | 40.14% | 40.14% | 40.14% | 9.86 pp | -29 | 12 | -2.42 |
| BTC Daily | transformer | Transformer | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | lstm | LSTM | 147 | 55 | 92 | 37.41% | 37.41% | 37.41% | 12.59 pp | -37 | 13 | -2.85 |
| BTC Hourly | rf | RandomForest | 123 | 51 | 72 | 41.46% | 41.46% | 41.46% | 8.54 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 123 | 46 | 77 | 37.40% | 37.40% | 37.40% | 12.60 pp | -31 | 6 | -5.17 |
| BTC Daily | xgb | XGBoost | 159 | 58 | 101 | 36.48% | 36.48% | 36.48% | 13.52 pp | -43 | 8 | -5.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |
| BTC Daily | lstm | LSTM | 149 | 52 | 97 | 34.90% | 34.90% | 34.90% | 15.10 pp | -45 | 7 | -6.43 |
| BTC Hourly | lstm | LSTM | 123 | 41 | 82 | 33.33% | 33.33% | 33.33% | 16.67 pp | -41 | 6 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 123 | 61 | 62 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 6 | -0.17 |
| BTC Hourly | transformer | Transformer | 123 | 60 | 63 | 48.78% | 48.78% | 48.78% | 1.22 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 123 | 57 | 66 | 46.34% | 46.34% | 46.34% | 3.66 pp | -9 | 6 | -1.50 |
| BTC Hourly | rf | RandomForest | 123 | 51 | 72 | 41.46% | 41.46% | 41.46% | 8.54 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 123 | 46 | 77 | 37.40% | 37.40% | 37.40% | 12.60 pp | -31 | 6 | -5.17 |
| BTC Hourly | lstm | LSTM | 123 | 41 | 82 | 33.33% | 33.33% | 33.33% | 16.67 pp | -41 | 6 | -6.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 149 | 73 | 76 | 48.99% | 48.99% | 48.99% | 1.01 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 149 | 67 | 82 | 44.97% | 44.97% | 44.97% | 5.03 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 159 | 58 | 101 | 36.48% | 36.48% | 36.48% | 13.52 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 149 | 52 | 97 | 34.90% | 34.90% | 34.90% | 15.10 pp | -45 | 7 | -6.43 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 147 | 77 | 70 | 52.38% | 52.38% | 52.38% | 2.38 pp | 7 | 12 | 0.58 |
| BTC Market Hours | rf | RandomForest | 147 | 69 | 78 | 46.94% | 46.94% | 46.94% | 3.06 pp | -9 | 12 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 12 | -1.25 |
| BTC Market Hours | transformer | Transformer | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 12 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 147 | 62 | 85 | 42.18% | 42.18% | 42.18% | 7.82 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 147 | 59 | 88 | 40.14% | 40.14% | 40.14% | 9.86 pp | -29 | 12 | -2.42 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 147 | 66 | 81 | 44.90% | 44.90% | 44.90% | 5.10 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 147 | 65 | 82 | 44.22% | 44.22% | 44.22% | 5.78 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 147 | 60 | 87 | 40.82% | 40.82% | 40.82% | 9.18 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 147 | 55 | 92 | 37.41% | 37.41% | 37.41% | 12.59 pp | -37 | 13 | -2.85 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
