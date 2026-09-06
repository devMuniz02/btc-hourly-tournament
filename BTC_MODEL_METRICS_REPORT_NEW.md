# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T06:22:11.960984+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 384 | 202 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 179 | 91 | 88 | 50.84% | 50.84% | 50.84% | 0.84 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 202 | 104 | 98 | 51.49% | 51.49% | 51.49% | 1.49 pp | 6 | 17 | 0.35 |
| BTC Market Hours | nn | NN | 203 | 104 | 99 | 51.23% | 51.23% | 51.23% | 1.23 pp | 5 | 16 | 0.31 |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 203 | 99 | 104 | 48.77% | 48.77% | 48.77% | 1.23 pp | -5 | 16 | -0.31 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 202 | 98 | 104 | 48.51% | 48.51% | 48.51% | 1.49 pp | -6 | 17 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 203 | 96 | 107 | 47.29% | 47.29% | 47.29% | 2.71 pp | -11 | 16 | -0.69 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 205 | 98 | 107 | 47.80% | 47.80% | 47.80% | 2.20 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 203 | 92 | 111 | 45.32% | 45.32% | 45.32% | 4.68 pp | -19 | 16 | -1.19 |
| BTC Market Hours Daily | rf | RandomForest | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 17 | -1.53 |
| BTC Hourly | transformer | Transformer | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 8 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 203 | 87 | 116 | 42.86% | 42.86% | 42.86% | 7.14 pp | -29 | 16 | -1.81 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 203 | 84 | 119 | 41.38% | 41.38% | 41.38% | 8.62 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 202 | 82 | 120 | 40.59% | 40.59% | 40.59% | 9.41 pp | -38 | 17 | -2.24 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| BTC Daily | nn | NN | 205 | 92 | 113 | 44.88% | 44.88% | 44.88% | 5.12 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 202 | 81 | 121 | 40.10% | 40.10% | 40.10% | 9.90 pp | -40 | 17 | -2.35 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
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
| BTC Market Hours Daily | transformer | Transformer | 202 | 104 | 98 | 51.49% | 51.49% | 51.49% | 1.49 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 202 | 98 | 104 | 48.51% | 48.51% | 48.51% | 1.49 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 202 | 82 | 120 | 40.59% | 40.59% | 40.59% | 9.41 pp | -38 | 17 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 202 | 81 | 121 | 40.10% | 40.10% | 40.10% | 9.90 pp | -40 | 17 | -2.35 |

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
