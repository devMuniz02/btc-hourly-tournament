# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T12:02:12.735525+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 226 | 166 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 262 | 202 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 359 | 190 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 359 | 190 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 163 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 163 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 163 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T15:00:00+00:00 | 164 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 190 | 99 | 91 | 52.11% | 52.11% | 52.11% | 2.11 pp | 8 | 16 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 190 | 92 | 98 | 48.42% | 48.42% | 48.42% | 1.58 pp | -6 | 16 | -0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | nn | NN | 190 | 89 | 101 | 46.84% | 46.84% | 46.84% | 3.16 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| Consolidated Hourly | xgb | XGBoost | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 190 | 84 | 106 | 44.21% | 44.21% | 44.21% | 5.79 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | nn | NN | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 190 | 78 | 112 | 41.05% | 41.05% | 41.05% | 8.95 pp | -34 | 16 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 190 | 77 | 113 | 40.53% | 40.53% | 40.53% | 9.47 pp | -36 | 16 | -2.25 |
| Consolidated Market Hours Daily | nn | NN | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 166 | 84 | 82 | 50.60% | 50.60% | 50.60% | 0.60 pp | 2 | 7 | 0.29 |
| BTC Hourly | transformer | Transformer | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 166 | 70 | 96 | 42.17% | 42.17% | 42.17% | 7.83 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 166 | 60 | 106 | 36.14% | 36.14% | 36.14% | 13.86 pp | -46 | 7 | -6.57 |
| BTC Hourly | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 192 | 85 | 107 | 44.27% | 44.27% | 44.27% | 5.73 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 192 | 75 | 117 | 39.06% | 39.06% | 39.06% | 10.94 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 202 | 74 | 128 | 36.63% | 36.63% | 36.63% | 13.37 pp | -54 | 10 | -5.40 |
| BTC Daily | lstm | LSTM | 192 | 66 | 126 | 34.38% | 34.38% | 34.38% | 15.62 pp | -60 | 9 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 190 | 81 | 109 | 42.63% | 42.63% | 42.63% | 7.37 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 190 | 99 | 91 | 52.11% | 52.11% | 52.11% | 2.11 pp | 8 | 16 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 190 | 92 | 98 | 48.42% | 48.42% | 48.42% | 1.58 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 190 | 89 | 101 | 46.84% | 46.84% | 46.84% | 3.16 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 190 | 84 | 106 | 44.21% | 44.21% | 44.21% | 5.79 pp | -22 | 16 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 190 | 78 | 112 | 41.05% | 41.05% | 41.05% | 8.95 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 190 | 77 | 113 | 40.53% | 40.53% | 40.53% | 9.47 pp | -36 | 16 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | nn | NN | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | lstm | LSTM | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
