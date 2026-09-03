# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T03:05:02.131489+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 188 | 128 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 223 | 163 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 294 | 151 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 294 | 151 | 143 | 0 |
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
| BTC Hourly | mlp_sklearn | MLPClassifier | 128 | 63 | 65 | 49.22% | 49.22% | 49.22% | 0.78 pp | -2 | 6 | -0.33 |
| BTC Hourly | transformer | Transformer | 128 | 63 | 65 | 49.22% | 49.22% | 49.22% | 0.78 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 13 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 153 | 74 | 79 | 48.37% | 48.37% | 48.37% | 1.63 pp | -5 | 7 | -0.71 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours | rf | RandomForest | 151 | 69 | 82 | 45.70% | 45.70% | 45.70% | 4.30 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| BTC Market Hours | transformer | Transformer | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 13 | -1.46 |
| BTC Market Hours Daily | xgb | XGBoost | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 13 | -1.92 |
| BTC Hourly | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 12 | -2.25 |
| Consolidated Hourly | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| BTC Daily | nn | NN | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 151 | 56 | 95 | 37.09% | 37.09% | 37.09% | 12.91 pp | -39 | 13 | -3.00 |
| BTC Daily | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 128 | 53 | 75 | 41.41% | 41.41% | 41.41% | 8.59 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 6 | -5.33 |
| BTC Daily | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 7 | -6.14 |
| BTC Hourly | lstm | LSTM | 128 | 44 | 84 | 34.38% | 34.38% | 34.38% | 15.62 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 128 | 63 | 65 | 49.22% | 49.22% | 49.22% | 0.78 pp | -2 | 6 | -0.33 |
| BTC Hourly | transformer | Transformer | 128 | 63 | 65 | 49.22% | 49.22% | 49.22% | 0.78 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Hourly | rf | RandomForest | 128 | 53 | 75 | 41.41% | 41.41% | 41.41% | 8.59 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 128 | 44 | 84 | 34.38% | 34.38% | 34.38% | 15.62 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 153 | 74 | 79 | 48.37% | 48.37% | 48.37% | 1.63 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 7 | -6.14 |

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
| BTC Market Hours Daily | transformer | Transformer | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 151 | 72 | 79 | 47.68% | 47.68% | 47.68% | 2.32 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 151 | 68 | 83 | 45.03% | 45.03% | 45.03% | 4.97 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 13 | -1.46 |
| BTC Market Hours Daily | xgb | XGBoost | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 151 | 56 | 95 | 37.09% | 37.09% | 37.09% | 12.91 pp | -39 | 13 | -3.00 |

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
