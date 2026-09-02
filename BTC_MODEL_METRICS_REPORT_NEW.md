# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T02:40:54.954126+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 171 | 111 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 207 | 147 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 265 | 135 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 265 | 135 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 135 | 71 | 64 | 52.59% | 52.59% | 52.59% | 2.59 pp | 7 | 11 | 0.64 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 7 | -0.43 |
| BTC Market Hours | rf | RandomForest | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 135 | 63 | 72 | 46.67% | 46.67% | 46.67% | 3.33 pp | -9 | 12 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 135 | 63 | 72 | 46.67% | 46.67% | 46.67% | 3.33 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 12 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| BTC Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | transformer | Transformer | 135 | 59 | 76 | 43.70% | 43.70% | 43.70% | 6.30 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 135 | 56 | 79 | 41.48% | 41.48% | 41.48% | 8.52 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Daily | transformer | Transformer | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 135 | 51 | 84 | 37.78% | 37.78% | 37.78% | 12.22 pp | -33 | 12 | -2.75 |
| BTC Market Hours | lstm | LSTM | 135 | 52 | 83 | 38.52% | 38.52% | 38.52% | 11.48 pp | -31 | 11 | -2.82 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 111 | 45 | 66 | 40.54% | 40.54% | 40.54% | 9.46 pp | -21 | 5 | -4.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 147 | 53 | 94 | 36.05% | 36.05% | 36.05% | 13.95 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 137 | 48 | 89 | 35.04% | 35.04% | 35.04% | 14.96 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 111 | 38 | 73 | 34.23% | 34.23% | 34.23% | 15.77 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 111 | 35 | 76 | 31.53% | 31.53% | 31.53% | 18.47 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 5 | -1.40 |
| BTC Hourly | nn | NN | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 5 | -1.80 |
| BTC Hourly | rf | RandomForest | 111 | 45 | 66 | 40.54% | 40.54% | 40.54% | 9.46 pp | -21 | 5 | -4.20 |
| BTC Hourly | xgb | XGBoost | 111 | 38 | 73 | 34.23% | 34.23% | 34.23% | 15.77 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 111 | 35 | 76 | 31.53% | 31.53% | 31.53% | 18.47 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 137 | 63 | 74 | 45.99% | 45.99% | 45.99% | 4.01 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 7 | -2.14 |
| BTC Daily | rf | RandomForest | 137 | 57 | 80 | 41.61% | 41.61% | 41.61% | 8.39 pp | -23 | 7 | -3.29 |
| BTC Daily | xgb | XGBoost | 147 | 53 | 94 | 36.05% | 36.05% | 36.05% | 13.95 pp | -41 | 8 | -5.12 |
| BTC Daily | lstm | LSTM | 137 | 48 | 89 | 35.04% | 35.04% | 35.04% | 14.96 pp | -41 | 7 | -5.86 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 135 | 71 | 64 | 52.59% | 52.59% | 52.59% | 2.59 pp | 7 | 11 | 0.64 |
| BTC Market Hours | rf | RandomForest | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 135 | 63 | 72 | 46.67% | 46.67% | 46.67% | 3.33 pp | -9 | 11 | -0.82 |
| BTC Market Hours | transformer | Transformer | 135 | 59 | 76 | 43.70% | 43.70% | 43.70% | 6.30 pp | -17 | 11 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 135 | 57 | 78 | 42.22% | 42.22% | 42.22% | 7.78 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 135 | 52 | 83 | 38.52% | 38.52% | 38.52% | 11.48 pp | -31 | 11 | -2.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 135 | 63 | 72 | 46.67% | 46.67% | 46.67% | 3.33 pp | -9 | 12 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | nn | NN | 135 | 61 | 74 | 45.19% | 45.19% | 45.19% | 4.81 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | xgb | XGBoost | 135 | 56 | 79 | 41.48% | 41.48% | 41.48% | 8.52 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 135 | 51 | 84 | 37.78% | 37.78% | 37.78% | 12.22 pp | -33 | 12 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
