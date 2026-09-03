# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T01:48:24.808852+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 187 | 127 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 223 | 163 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 294 | 151 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 293 | 150 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 127 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 127 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 24 | 103 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 24 | 103 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| BTC Market Hours | nn | NN | 151 | 78 | 73 | 51.66% | 51.66% | 51.66% | 1.66 pp | 5 | 12 | 0.42 |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| BTC Hourly | transformer | Transformer | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 7 | -0.43 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 127 | 62 | 65 | 48.82% | 48.82% | 48.82% | 1.18 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 13 | -1.08 |
| BTC Market Hours | rf | RandomForest | 151 | 69 | 82 | 45.70% | 45.70% | 45.70% | 4.30 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| BTC Market Hours | transformer | Transformer | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 150 | 66 | 84 | 44.00% | 44.00% | 44.00% | 6.00 pp | -18 | 13 | -1.38 |
| BTC Hourly | nn | NN | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 13 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 12 | -2.25 |
| Consolidated Hourly | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| BTC Daily | nn | NN | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | lstm | LSTM | 150 | 56 | 94 | 37.33% | 37.33% | 37.33% | 12.67 pp | -38 | 13 | -2.92 |
| BTC Daily | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 6 | -3.83 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 6 | -5.50 |
| BTC Daily | xgb | XGBoost | 163 | 58 | 105 | 35.58% | 35.58% | 35.58% | 14.42 pp | -47 | 8 | -5.88 |
| BTC Daily | lstm | LSTM | 153 | 54 | 99 | 35.29% | 35.29% | 35.29% | 14.71 pp | -45 | 7 | -6.43 |
| BTC Hourly | lstm | LSTM | 127 | 43 | 84 | 33.86% | 33.86% | 33.86% | 16.14 pp | -41 | 6 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 6 | -0.17 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 127 | 62 | 65 | 48.82% | 48.82% | 48.82% | 1.18 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 6 | -1.83 |
| BTC Hourly | rf | RandomForest | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 6 | -3.83 |
| BTC Hourly | xgb | XGBoost | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 127 | 43 | 84 | 33.86% | 33.86% | 33.86% | 16.14 pp | -41 | 6 | -6.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 163 | 58 | 105 | 35.58% | 35.58% | 35.58% | 14.42 pp | -47 | 8 | -5.88 |
| BTC Daily | lstm | LSTM | 153 | 54 | 99 | 35.29% | 35.29% | 35.29% | 14.71 pp | -45 | 7 | -6.43 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 151 | 78 | 73 | 51.66% | 51.66% | 51.66% | 1.66 pp | 5 | 12 | 0.42 |
| BTC Market Hours | rf | RandomForest | 151 | 69 | 82 | 45.70% | 45.70% | 45.70% | 4.30 pp | -13 | 12 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| BTC Market Hours | transformer | Transformer | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 12 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | nn | NN | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 150 | 66 | 84 | 44.00% | 44.00% | 44.00% | 6.00 pp | -18 | 13 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 150 | 56 | 94 | 37.33% | 37.33% | 37.33% | 12.67 pp | -38 | 13 | -2.92 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
