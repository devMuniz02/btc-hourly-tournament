# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T07:03:40.895785+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 239 | 179 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 275 | 215 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 385 | 203 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 385 | 203 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 176 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 203 | 105 | 98 | 51.72% | 51.72% | 51.72% | 1.72 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 179 | 91 | 88 | 50.84% | 50.84% | 50.84% | 0.84 pp | 3 | 8 | 0.38 |
| BTC Market Hours | nn | NN | 203 | 104 | 99 | 51.23% | 51.23% | 51.23% | 1.23 pp | 5 | 16 | 0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 17 | -0.29 |
| BTC Market Hours | transformer | Transformer | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 16 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 17 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 203 | 96 | 107 | 47.29% | 47.29% | 47.29% | 2.71 pp | -11 | 16 | -0.69 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 203 | 92 | 111 | 45.32% | 45.32% | 45.32% | 4.68 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 17 | -1.47 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| BTC Hourly | transformer | Transformer | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 16 | -1.81 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 203 | 83 | 120 | 40.89% | 40.89% | 40.89% | 9.11 pp | -37 | 17 | -2.18 |
| BTC Market Hours | lstm | LSTM | 203 | 84 | 119 | 41.38% | 41.38% | 41.38% | 8.62 pp | -35 | 16 | -2.19 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Daily | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 203 | 81 | 122 | 39.90% | 39.90% | 39.90% | 10.10 pp | -41 | 17 | -2.41 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 205 | 86 | 119 | 41.95% | 41.95% | 41.95% | 8.05 pp | -33 | 9 | -3.67 |
| BTC Hourly | xgb | XGBoost | 179 | 67 | 112 | 37.43% | 37.43% | 37.43% | 12.57 pp | -45 | 8 | -5.62 |
| BTC Daily | rf | RandomForest | 205 | 77 | 128 | 37.56% | 37.56% | 37.56% | 12.44 pp | -51 | 9 | -5.67 |
| BTC Hourly | lstm | LSTM | 179 | 66 | 113 | 36.87% | 36.87% | 36.87% | 13.13 pp | -47 | 8 | -5.88 |
| BTC Daily | xgb | XGBoost | 215 | 76 | 139 | 35.35% | 35.35% | 35.35% | 14.65 pp | -63 | 10 | -6.30 |
| BTC Daily | lstm | LSTM | 205 | 68 | 137 | 33.17% | 33.17% | 33.17% | 16.83 pp | -69 | 9 | -7.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 179 | 91 | 88 | 50.84% | 50.84% | 50.84% | 0.84 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 8 | -1.62 |
| BTC Hourly | nn | NN | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 179 | 67 | 112 | 37.43% | 37.43% | 37.43% | 12.57 pp | -45 | 8 | -5.62 |
| BTC Hourly | lstm | LSTM | 179 | 66 | 113 | 36.87% | 36.87% | 36.87% | 13.13 pp | -47 | 8 | -5.88 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 9 | -1.00 |
| BTC Daily | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Daily | transformer | Transformer | 205 | 86 | 119 | 41.95% | 41.95% | 41.95% | 8.05 pp | -33 | 9 | -3.67 |
| BTC Daily | rf | RandomForest | 205 | 77 | 128 | 37.56% | 37.56% | 37.56% | 12.44 pp | -51 | 9 | -5.67 |
| BTC Daily | xgb | XGBoost | 215 | 76 | 139 | 35.35% | 35.35% | 35.35% | 14.65 pp | -63 | 10 | -6.30 |
| BTC Daily | lstm | LSTM | 205 | 68 | 137 | 33.17% | 33.17% | 33.17% | 16.83 pp | -69 | 9 | -7.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 203 | 104 | 99 | 51.23% | 51.23% | 51.23% | 1.23 pp | 5 | 16 | 0.31 |
| BTC Market Hours | transformer | Transformer | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 16 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 203 | 96 | 107 | 47.29% | 47.29% | 47.29% | 2.71 pp | -11 | 16 | -0.69 |
| BTC Market Hours | rf | RandomForest | 203 | 92 | 111 | 45.32% | 45.32% | 45.32% | 4.68 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 16 | -1.81 |
| BTC Market Hours | lstm | LSTM | 203 | 84 | 119 | 41.38% | 41.38% | 41.38% | 8.62 pp | -35 | 16 | -2.19 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 203 | 105 | 98 | 51.72% | 51.72% | 51.72% | 1.72 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 17 | -0.29 |
| BTC Market Hours Daily | nn | NN | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 203 | 89 | 114 | 43.84% | 43.84% | 43.84% | 6.16 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 203 | 83 | 120 | 40.89% | 40.89% | 40.89% | 9.11 pp | -37 | 17 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 203 | 81 | 122 | 39.90% | 39.90% | 39.90% | 10.10 pp | -41 | 17 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
