# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T17:50:56.186371+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 213 | 153 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 249 | 189 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 16:00:00+00:00 | 338 | 177 | 161 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 16:00:00+00:00 | 338 | 177 | 161 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 151 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 21:00:00+00:00 | 151 | 37 | 114 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 153 | 81 | 72 | 52.94% | 52.94% | 52.94% | 2.94 pp | 9 | 7 | 1.29 |
| BTC Market Hours | nn | NN | 177 | 93 | 84 | 52.54% | 52.54% | 52.54% | 2.54 pp | 9 | 14 | 0.64 |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| BTC Market Hours Daily | transformer | Transformer | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 15 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 15 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| BTC Market Hours | transformer | Transformer | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 14 | -0.50 |
| BTC Hourly | transformer | Transformer | 153 | 74 | 79 | 48.37% | 48.37% | 48.37% | 1.63 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 177 | 83 | 94 | 46.89% | 46.89% | 46.89% | 3.11 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours | rf | RandomForest | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 14 | -1.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | rf | RandomForest | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 15 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 15 | -2.07 |
| BTC Market Hours | lstm | LSTM | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 177 | 70 | 107 | 39.55% | 39.55% | 39.55% | 10.45 pp | -37 | 15 | -2.47 |
| BTC Daily | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 8 | -2.88 |
| BTC Hourly | nn | NN | 153 | 66 | 87 | 43.14% | 43.14% | 43.14% | 6.86 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 179 | 73 | 106 | 40.78% | 40.78% | 40.78% | 9.22 pp | -33 | 8 | -4.12 |
| BTC Daily | xgb | XGBoost | 189 | 69 | 120 | 36.51% | 36.51% | 36.51% | 13.49 pp | -51 | 9 | -5.67 |
| BTC Hourly | lstm | LSTM | 153 | 56 | 97 | 36.60% | 36.60% | 36.60% | 13.40 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 179 | 63 | 116 | 35.20% | 35.20% | 35.20% | 14.80 pp | -53 | 8 | -6.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 153 | 81 | 72 | 52.94% | 52.94% | 52.94% | 2.94 pp | 9 | 7 | 1.29 |
| BTC Hourly | transformer | Transformer | 153 | 74 | 79 | 48.37% | 48.37% | 48.37% | 1.63 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 153 | 66 | 87 | 43.14% | 43.14% | 43.14% | 6.86 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 153 | 56 | 97 | 36.60% | 36.60% | 36.60% | 13.40 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 7 | -6.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 8 | -1.62 |
| BTC Daily | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 8 | -2.62 |
| BTC Daily | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 179 | 73 | 106 | 40.78% | 40.78% | 40.78% | 9.22 pp | -33 | 8 | -4.12 |
| BTC Daily | xgb | XGBoost | 189 | 69 | 120 | 36.51% | 36.51% | 36.51% | 13.49 pp | -51 | 9 | -5.67 |
| BTC Daily | lstm | LSTM | 179 | 63 | 116 | 35.20% | 35.20% | 35.20% | 14.80 pp | -53 | 8 | -6.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 177 | 93 | 84 | 52.54% | 52.54% | 52.54% | 2.54 pp | 9 | 14 | 0.64 |
| BTC Market Hours | transformer | Transformer | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 14 | -0.50 |
| BTC Market Hours | rf | RandomForest | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 14 | -1.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 14 | -1.21 |
| BTC Market Hours | xgb | XGBoost | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 14 | -2.21 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 15 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 15 | -0.20 |
| BTC Market Hours Daily | nn | NN | 177 | 83 | 94 | 46.89% | 46.89% | 46.89% | 3.11 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 177 | 78 | 99 | 44.07% | 44.07% | 44.07% | 5.93 pp | -21 | 15 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 177 | 73 | 104 | 41.24% | 41.24% | 41.24% | 8.76 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 177 | 70 | 107 | 39.55% | 39.55% | 39.55% | 10.45 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 151 | 77 | 74 | 50.99% | 50.99% | 50.99% | 0.99 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 151 | 70 | 81 | 46.36% | 46.36% | 46.36% | 3.64 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 151 | 65 | 86 | 43.05% | 43.05% | 43.05% | 6.95 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
