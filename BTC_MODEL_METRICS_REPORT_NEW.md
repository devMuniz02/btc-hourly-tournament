# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T13:41:51.313309+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 179 | 119 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 214 | 154 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 12:00:00+00:00 | 273 | 142 | 131 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 12:00:00+00:00 | 273 | 142 | 131 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 17:00:00+00:00 | 119 | 119 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 17:00:00+00:00 | 119 | 119 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 17:00:00+00:00 | 119 | 20 | 99 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 17:00:00+00:00 | 119 | 20 | 99 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 119 | 62 | 57 | 52.10% | 52.10% | 52.10% | 2.10 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 62 | 57 | 52.10% | 52.10% | 52.10% | 2.10 pp | 5 | 10 | 0.50 |
| BTC Market Hours | nn | NN | 142 | 73 | 69 | 51.41% | 51.41% | 51.41% | 1.41 pp | 4 | 11 | 0.36 |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 119 | 59 | 60 | 49.58% | 49.58% | 49.58% | 0.42 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 7 | -0.57 |
| BTC Hourly | transformer | Transformer | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 142 | 67 | 75 | 47.18% | 47.18% | 47.18% | 2.82 pp | -8 | 12 | -0.67 |
| BTC Market Hours | rf | RandomForest | 142 | 66 | 76 | 46.48% | 46.48% | 46.48% | 3.52 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | transformer | Transformer | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Hourly | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Hourly | nn | NN | 119 | 49 | 70 | 41.18% | 41.18% | 41.18% | 8.82 pp | -21 | 10 | -2.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 49 | 70 | 41.18% | 41.18% | 41.18% | 8.82 pp | -21 | 10 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 12 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 11 | -2.18 |
| BTC Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 7 | -2.29 |
| BTC Market Hours | lstm | LSTM | 142 | 57 | 85 | 40.14% | 40.14% | 40.14% | 9.86 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 142 | 54 | 88 | 38.03% | 38.03% | 38.03% | 11.97 pp | -34 | 12 | -2.83 |
| BTC Daily | transformer | Transformer | 144 | 62 | 82 | 43.06% | 43.06% | 43.06% | 6.94 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| BTC Daily | xgb | XGBoost | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 7 | -5.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 119 | 39 | 80 | 32.77% | 32.77% | 32.77% | 17.23 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 119 | 59 | 60 | 49.58% | 49.58% | 49.58% | 0.42 pp | -1 | 5 | -0.20 |
| BTC Hourly | transformer | Transformer | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 5 | -0.60 |
| BTC Hourly | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 5 | -1.80 |
| BTC Hourly | rf | RandomForest | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 119 | 39 | 80 | 32.77% | 32.77% | 32.77% | 17.23 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 144 | 70 | 74 | 48.61% | 48.61% | 48.61% | 1.39 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 144 | 64 | 80 | 44.44% | 44.44% | 44.44% | 5.56 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 144 | 62 | 82 | 43.06% | 43.06% | 43.06% | 6.94 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 144 | 60 | 84 | 41.67% | 41.67% | 41.67% | 8.33 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 154 | 56 | 98 | 36.36% | 36.36% | 36.36% | 13.64 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 144 | 52 | 92 | 36.11% | 36.11% | 36.11% | 13.89 pp | -40 | 7 | -5.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 142 | 73 | 69 | 51.41% | 51.41% | 51.41% | 1.41 pp | 4 | 11 | 0.36 |
| BTC Market Hours | rf | RandomForest | 142 | 66 | 76 | 46.48% | 46.48% | 46.48% | 3.52 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 142 | 65 | 77 | 45.77% | 45.77% | 45.77% | 4.23 pp | -12 | 11 | -1.09 |
| BTC Market Hours | xgb | XGBoost | 142 | 59 | 83 | 41.55% | 41.55% | 41.55% | 8.45 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 142 | 57 | 85 | 40.14% | 40.14% | 40.14% | 9.86 pp | -28 | 11 | -2.55 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 142 | 69 | 73 | 48.59% | 48.59% | 48.59% | 1.41 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 142 | 67 | 75 | 47.18% | 47.18% | 47.18% | 2.82 pp | -8 | 12 | -0.67 |
| BTC Market Hours Daily | nn | NN | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 142 | 63 | 79 | 44.37% | 44.37% | 44.37% | 5.63 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 142 | 58 | 84 | 40.85% | 40.85% | 40.85% | 9.15 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 142 | 54 | 88 | 38.03% | 38.03% | 38.03% | 11.97 pp | -34 | 12 | -2.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 119 | 62 | 57 | 52.10% | 52.10% | 52.10% | 2.10 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 119 | 49 | 70 | 41.18% | 41.18% | 41.18% | 8.82 pp | -21 | 10 | -2.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 119 | 62 | 57 | 52.10% | 52.10% | 52.10% | 2.10 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 119 | 57 | 62 | 47.90% | 47.90% | 47.90% | 2.10 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 119 | 54 | 65 | 45.38% | 45.38% | 45.38% | 4.62 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 119 | 49 | 70 | 41.18% | 41.18% | 41.18% | 8.82 pp | -21 | 10 | -2.10 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
