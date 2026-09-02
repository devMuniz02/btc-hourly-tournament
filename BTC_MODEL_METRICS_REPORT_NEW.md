# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T01:23:25.022995+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 170 | 110 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 206 | 146 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 264 | 134 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 264 | 134 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 111 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 134 | 71 | 63 | 52.99% | 52.99% | 52.99% | 2.99 pp | 8 | 11 | 0.73 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 56 | 55 | 50.45% | 50.45% | 50.45% | 0.45 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 12 | -0.33 |
| BTC Market Hours | rf | RandomForest | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 6 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| Consolidated Hourly | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 10 | -0.90 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 50 | 61 | 45.05% | 45.05% | 45.05% | 4.95 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 12 | -1.17 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 5 | -1.60 |
| BTC Daily | nn | NN | 136 | 63 | 73 | 46.32% | 46.32% | 46.32% | 3.68 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 11 | -1.82 |
| BTC Market Hours Daily | xgb | XGBoost | 134 | 55 | 79 | 41.04% | 41.04% | 41.04% | 8.96 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Daily | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 6 | -2.67 |
| BTC Market Hours Daily | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 12 | -2.67 |
| BTC Market Hours | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 136 | 57 | 79 | 41.91% | 41.91% | 41.91% | 8.09 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 5 | -4.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |
| BTC Daily | lstm | LSTM | 136 | 48 | 88 | 35.29% | 35.29% | 35.29% | 14.71 pp | -40 | 6 | -6.67 |
| BTC Hourly | xgb | XGBoost | 110 | 38 | 72 | 34.55% | 34.55% | 34.55% | 15.45 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 110 | 34 | 76 | 30.91% | 30.91% | 30.91% | 19.09 pp | -42 | 5 | -8.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Hourly | nn | NN | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 5 | -4.40 |
| BTC Hourly | xgb | XGBoost | 110 | 38 | 72 | 34.55% | 34.55% | 34.55% | 15.45 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 110 | 34 | 76 | 30.91% | 30.91% | 30.91% | 19.09 pp | -42 | 5 | -8.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 6 | -0.67 |
| BTC Daily | nn | NN | 136 | 63 | 73 | 46.32% | 46.32% | 46.32% | 3.68 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 6 | -2.67 |
| BTC Daily | rf | RandomForest | 136 | 57 | 79 | 41.91% | 41.91% | 41.91% | 8.09 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 146 | 53 | 93 | 36.30% | 36.30% | 36.30% | 13.70 pp | -40 | 7 | -5.71 |
| BTC Daily | lstm | LSTM | 136 | 48 | 88 | 35.29% | 35.29% | 35.29% | 14.71 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 134 | 71 | 63 | 52.99% | 52.99% | 52.99% | 2.99 pp | 8 | 11 | 0.73 |
| BTC Market Hours | rf | RandomForest | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| BTC Market Hours | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 11 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 11 | -1.82 |
| BTC Market Hours | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 11 | -2.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | rf | RandomForest | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 134 | 55 | 79 | 41.04% | 41.04% | 41.04% | 8.96 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 12 | -2.67 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
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
