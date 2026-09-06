# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T18:01:51.075373+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 247 | 187 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 282 | 222 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 398 | 210 | 188 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 398 | 210 | 188 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 17 | 0.47 |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| BTC Market Hours | transformer | Transformer | 210 | 103 | 107 | 49.05% | 49.05% | 49.05% | 0.95 pp | -4 | 17 | -0.24 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 210 | 99 | 111 | 47.14% | 47.14% | 47.14% | 2.86 pp | -12 | 17 | -0.71 |
| BTC Market Hours | rf | RandomForest | 210 | 97 | 113 | 46.19% | 46.19% | 46.19% | 3.81 pp | -16 | 17 | -0.94 |
| BTC Daily | mlp_sklearn | MLPClassifier | 212 | 101 | 111 | 47.64% | 47.64% | 47.64% | 2.36 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 17 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| BTC Hourly | transformer | Transformer | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 17 | -2.12 |
| BTC Daily | nn | NN | 212 | 95 | 117 | 44.81% | 44.81% | 44.81% | 5.19 pp | -22 | 10 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Hourly | nn | NN | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 8 | -3.88 |
| BTC Daily | rf | RandomForest | 212 | 80 | 132 | 37.74% | 37.74% | 37.74% | 12.26 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 222 | 79 | 143 | 35.59% | 35.59% | 35.59% | 14.41 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 8 | -6.38 |
| BTC Daily | lstm | LSTM | 212 | 71 | 141 | 33.49% | 33.49% | 33.49% | 16.51 pp | -70 | 10 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 8 | -1.88 |
| BTC Hourly | nn | NN | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 8 | -3.88 |
| BTC Hourly | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 8 | -6.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 212 | 101 | 111 | 47.64% | 47.64% | 47.64% | 2.36 pp | -10 | 10 | -1.00 |
| BTC Daily | nn | NN | 212 | 95 | 117 | 44.81% | 44.81% | 44.81% | 5.19 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 212 | 88 | 124 | 41.51% | 41.51% | 41.51% | 8.49 pp | -36 | 10 | -3.60 |
| BTC Daily | rf | RandomForest | 212 | 80 | 132 | 37.74% | 37.74% | 37.74% | 12.26 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 222 | 79 | 143 | 35.59% | 35.59% | 35.59% | 14.41 pp | -64 | 11 | -5.82 |
| BTC Daily | lstm | LSTM | 212 | 71 | 141 | 33.49% | 33.49% | 33.49% | 16.51 pp | -70 | 10 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 17 | 0.47 |
| BTC Market Hours | transformer | Transformer | 210 | 103 | 107 | 49.05% | 49.05% | 49.05% | 0.95 pp | -4 | 17 | -0.24 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 210 | 99 | 111 | 47.14% | 47.14% | 47.14% | 2.86 pp | -12 | 17 | -0.71 |
| BTC Market Hours | rf | RandomForest | 210 | 97 | 113 | 46.19% | 46.19% | 46.19% | 3.81 pp | -16 | 17 | -0.94 |
| BTC Market Hours | xgb | XGBoost | 210 | 92 | 118 | 43.81% | 43.81% | 43.81% | 6.19 pp | -26 | 17 | -1.53 |
| BTC Market Hours | lstm | LSTM | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 17 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
