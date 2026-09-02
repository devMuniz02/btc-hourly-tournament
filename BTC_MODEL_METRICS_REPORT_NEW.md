# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T00:40:57.642011+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 263 | 134 | 129 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 262 | 133 | 129 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 134 | 71 | 63 | 52.99% | 52.99% | 52.99% | 2.99 pp | 8 | 11 | 0.73 |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| BTC Market Hours | rf | RandomForest | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 136 | 66 | 70 | 48.53% | 48.53% | 48.53% | 1.47 pp | -4 | 6 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | rf | RandomForest | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 11 | -1.45 |
| BTC Hourly | nn | NN | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 5 | -1.60 |
| BTC Daily | nn | NN | 136 | 63 | 73 | 46.32% | 46.32% | 46.32% | 3.68 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 11 | -1.82 |
| BTC Market Hours Daily | xgb | XGBoost | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 12 | -2.58 |
| BTC Daily | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 6 | -2.67 |
| BTC Market Hours | lstm | LSTM | 134 | 51 | 83 | 38.06% | 38.06% | 38.06% | 11.94 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 136 | 57 | 79 | 41.91% | 41.91% | 41.91% | 8.09 pp | -22 | 6 | -3.67 |
| BTC Hourly | rf | RandomForest | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 5 | -4.40 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | rf | RandomForest | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | xgb | XGBoost | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 12 | -2.58 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
