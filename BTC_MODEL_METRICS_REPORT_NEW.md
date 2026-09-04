# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T21:00:40.748971+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 216 | 156 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 252 | 192 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 345 | 180 | 165 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 20:00:00+00:00 | 345 | 180 | 165 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 7 | 1.14 |
| BTC Market Hours | nn | NN | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 14 | 0.43 |
| BTC Market Hours Daily | transformer | Transformer | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 15 | 0.40 |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 15 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| BTC Market Hours | transformer | Transformer | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | nn | NN | 180 | 85 | 95 | 47.22% | 47.22% | 47.22% | 2.78 pp | -10 | 15 | -0.67 |
| BTC Hourly | transformer | Transformer | 156 | 75 | 81 | 48.08% | 48.08% | 48.08% | 1.92 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | rf | RandomForest | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 15 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 14 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 15 | -2.00 |
| BTC Market Hours | lstm | LSTM | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| BTC Daily | nn | NN | 182 | 81 | 101 | 44.51% | 44.51% | 44.51% | 5.49 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 180 | 71 | 109 | 39.44% | 39.44% | 39.44% | 10.56 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Daily | transformer | Transformer | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 8 | -2.75 |
| BTC Hourly | nn | NN | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 9 | -5.78 |
| BTC Hourly | lstm | LSTM | 156 | 57 | 99 | 36.54% | 36.54% | 36.54% | 13.46 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 156 | 56 | 100 | 35.90% | 35.90% | 35.90% | 14.10 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 182 | 63 | 119 | 34.62% | 34.62% | 34.62% | 15.38 pp | -56 | 8 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 156 | 82 | 74 | 52.56% | 52.56% | 52.56% | 2.56 pp | 8 | 7 | 1.14 |
| BTC Hourly | transformer | Transformer | 156 | 75 | 81 | 48.08% | 48.08% | 48.08% | 1.92 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 156 | 67 | 89 | 42.95% | 42.95% | 42.95% | 7.05 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 156 | 57 | 99 | 36.54% | 36.54% | 36.54% | 13.46 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 156 | 56 | 100 | 35.90% | 35.90% | 35.90% | 14.10 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 8 | -1.25 |
| BTC Daily | nn | NN | 182 | 81 | 101 | 44.51% | 44.51% | 44.51% | 5.49 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 192 | 70 | 122 | 36.46% | 36.46% | 36.46% | 13.54 pp | -52 | 9 | -5.78 |
| BTC Daily | lstm | LSTM | 182 | 63 | 119 | 34.62% | 34.62% | 34.62% | 15.38 pp | -56 | 8 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 14 | 0.43 |
| BTC Market Hours | transformer | Transformer | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 14 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | rf | RandomForest | 180 | 82 | 98 | 45.56% | 45.56% | 45.56% | 4.44 pp | -16 | 14 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 180 | 77 | 103 | 42.78% | 42.78% | 42.78% | 7.22 pp | -26 | 14 | -1.86 |
| BTC Market Hours | lstm | LSTM | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 180 | 93 | 87 | 51.67% | 51.67% | 51.67% | 1.67 pp | 6 | 15 | 0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 180 | 88 | 92 | 48.89% | 48.89% | 48.89% | 1.11 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 180 | 85 | 95 | 47.22% | 47.22% | 47.22% | 2.78 pp | -10 | 15 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 15 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 180 | 75 | 105 | 41.67% | 41.67% | 41.67% | 8.33 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 180 | 71 | 109 | 39.44% | 39.44% | 39.44% | 10.56 pp | -38 | 15 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
