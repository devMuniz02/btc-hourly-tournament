# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T08:01:17.077521+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 240 | 180 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 276 | 216 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 386 | 204 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 386 | 204 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 180 | 92 | 88 | 51.11% | 51.11% | 51.11% | 1.11 pp | 4 | 8 | 0.50 |
| BTC Market Hours | nn | NN | 204 | 105 | 99 | 51.47% | 51.47% | 51.47% | 1.47 pp | 6 | 16 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 204 | 105 | 99 | 51.47% | 51.47% | 51.47% | 1.47 pp | 6 | 17 | 0.35 |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 16 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 204 | 99 | 105 | 48.53% | 48.53% | 48.53% | 1.47 pp | -6 | 17 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 204 | 96 | 108 | 47.06% | 47.06% | 47.06% | 2.94 pp | -12 | 16 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 206 | 99 | 107 | 48.06% | 48.06% | 48.06% | 1.94 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 204 | 93 | 111 | 45.59% | 45.59% | 45.59% | 4.41 pp | -18 | 16 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 17 | -1.53 |
| BTC Hourly | transformer | Transformer | 180 | 83 | 97 | 46.11% | 46.11% | 46.11% | 3.89 pp | -14 | 8 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 204 | 88 | 116 | 43.14% | 43.14% | 43.14% | 6.86 pp | -28 | 16 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 204 | 83 | 121 | 40.69% | 40.69% | 40.69% | 9.31 pp | -38 | 17 | -2.24 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| BTC Daily | nn | NN | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | lstm | LSTM | 204 | 81 | 123 | 39.71% | 39.71% | 39.71% | 10.29 pp | -42 | 17 | -2.47 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 206 | 86 | 120 | 41.75% | 41.75% | 41.75% | 8.25 pp | -34 | 9 | -3.78 |
| BTC Hourly | xgb | XGBoost | 180 | 68 | 112 | 37.78% | 37.78% | 37.78% | 12.22 pp | -44 | 8 | -5.50 |
| BTC Hourly | lstm | LSTM | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 8 | -5.75 |
| BTC Daily | rf | RandomForest | 206 | 77 | 129 | 37.38% | 37.38% | 37.38% | 12.62 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 206 | 69 | 137 | 33.50% | 33.50% | 33.50% | 16.50 pp | -68 | 9 | -7.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 180 | 92 | 88 | 51.11% | 51.11% | 51.11% | 1.11 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 180 | 83 | 97 | 46.11% | 46.11% | 46.11% | 3.89 pp | -14 | 8 | -1.75 |
| BTC Hourly | nn | NN | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 8 | -3.25 |
| BTC Hourly | xgb | XGBoost | 180 | 68 | 112 | 37.78% | 37.78% | 37.78% | 12.22 pp | -44 | 8 | -5.50 |
| BTC Hourly | lstm | LSTM | 180 | 67 | 113 | 37.22% | 37.22% | 37.22% | 12.78 pp | -46 | 8 | -5.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 206 | 99 | 107 | 48.06% | 48.06% | 48.06% | 1.94 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 206 | 92 | 114 | 44.66% | 44.66% | 44.66% | 5.34 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 206 | 86 | 120 | 41.75% | 41.75% | 41.75% | 8.25 pp | -34 | 9 | -3.78 |
| BTC Daily | rf | RandomForest | 206 | 77 | 129 | 37.38% | 37.38% | 37.38% | 12.62 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 206 | 69 | 137 | 33.50% | 33.50% | 33.50% | 16.50 pp | -68 | 9 | -7.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 204 | 105 | 99 | 51.47% | 51.47% | 51.47% | 1.47 pp | 6 | 16 | 0.38 |
| BTC Market Hours | transformer | Transformer | 204 | 100 | 104 | 49.02% | 49.02% | 49.02% | 0.98 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 204 | 96 | 108 | 47.06% | 47.06% | 47.06% | 2.94 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 204 | 93 | 111 | 45.59% | 45.59% | 45.59% | 4.41 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 204 | 88 | 116 | 43.14% | 43.14% | 43.14% | 6.86 pp | -28 | 16 | -1.75 |
| BTC Market Hours | lstm | LSTM | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 16 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 204 | 105 | 99 | 51.47% | 51.47% | 51.47% | 1.47 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 204 | 99 | 105 | 48.53% | 48.53% | 48.53% | 1.47 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 204 | 89 | 115 | 43.63% | 43.63% | 43.63% | 6.37 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 204 | 83 | 121 | 40.69% | 40.69% | 40.69% | 9.31 pp | -38 | 17 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 204 | 81 | 123 | 39.71% | 39.71% | 39.71% | 10.29 pp | -42 | 17 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
