# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T02:00:10.953488+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 206 | 146 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 264 | 134 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 264 | 134 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 111 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 111 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 16 | 95 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 16 | 95 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 134 | 71 | 63 | 52.99% | 52.99% | 52.99% | 2.99 pp | 8 | 11 | 0.73 |
| Consolidated Hourly | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 12 | -0.33 |
| BTC Market Hours | rf | RandomForest | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 7 | -0.57 |
| Consolidated Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| BTC Market Hours Daily | rf | RandomForest | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| BTC Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 11 | -1.45 |
| BTC Daily | nn | NN | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 7 | -1.71 |
| BTC Hourly | nn | NN | 111 | 51 | 60 | 45.95% | 45.95% | 45.95% | 4.05 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 134 | 55 | 79 | 41.04% | 41.04% | 41.04% | 8.96 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Daily | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 12 | -2.67 |
| BTC Market Hours | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 136 | 56 | 80 | 41.18% | 41.18% | 41.18% | 8.82 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 111 | 45 | 66 | 40.54% | 40.54% | 40.54% | 9.46 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 146 | 52 | 94 | 35.62% | 35.62% | 35.62% | 14.38 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 136 | 48 | 88 | 35.29% | 35.29% | 35.29% | 14.71 pp | -40 | 7 | -5.71 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 7 | -1.71 |
| BTC Daily | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 136 | 56 | 80 | 41.18% | 41.18% | 41.18% | 8.82 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 146 | 52 | 94 | 35.62% | 35.62% | 35.62% | 14.38 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 136 | 48 | 88 | 35.29% | 35.29% | 35.29% | 14.71 pp | -40 | 7 | -5.71 |

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
| Consolidated Hourly | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
