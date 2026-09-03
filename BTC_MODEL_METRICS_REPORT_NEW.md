# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T16:50:07.854471+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 197 | 137 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 233 | 173 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 308 | 161 | 147 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 15:00:00+00:00 | 308 | 161 | 147 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 135 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T13:00:00+00:00 | 136 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 6 | 0.50 |
| BTC Market Hours | nn | NN | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 13 | 0.38 |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 137 | 68 | 69 | 49.64% | 49.64% | 49.64% | 0.36 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 14 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| BTC Daily | nn | NN | 163 | 76 | 87 | 46.63% | 46.63% | 46.63% | 3.37 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 14 | -2.07 |
| BTC Market Hours | lstm | LSTM | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Daily | transformer | Transformer | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 8 | -2.38 |
| BTC Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 6 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 161 | 62 | 99 | 38.51% | 38.51% | 38.51% | 11.49 pp | -37 | 14 | -2.64 |
| BTC Daily | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 173 | 64 | 109 | 36.99% | 36.99% | 36.99% | 13.01 pp | -45 | 9 | -5.00 |
| BTC Daily | lstm | LSTM | 163 | 61 | 102 | 37.42% | 37.42% | 37.42% | 12.58 pp | -41 | 8 | -5.12 |
| BTC Hourly | xgb | XGBoost | 137 | 51 | 86 | 37.23% | 37.23% | 37.23% | 12.77 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 137 | 49 | 88 | 35.77% | 35.77% | 35.77% | 14.23 pp | -39 | 6 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 137 | 70 | 67 | 51.09% | 51.09% | 51.09% | 1.09 pp | 3 | 6 | 0.50 |
| BTC Hourly | transformer | Transformer | 137 | 68 | 69 | 49.64% | 49.64% | 49.64% | 0.36 pp | -1 | 6 | -0.17 |
| BTC Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 6 | -2.50 |
| BTC Hourly | rf | RandomForest | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 137 | 51 | 86 | 37.23% | 37.23% | 37.23% | 12.77 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 137 | 49 | 88 | 35.77% | 35.77% | 35.77% | 14.23 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 8 | -0.62 |
| BTC Daily | nn | NN | 163 | 76 | 87 | 46.63% | 46.63% | 46.63% | 3.37 pp | -11 | 8 | -1.38 |
| BTC Daily | transformer | Transformer | 163 | 72 | 91 | 44.17% | 44.17% | 44.17% | 5.83 pp | -19 | 8 | -2.38 |
| BTC Daily | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 8 | -3.12 |
| BTC Daily | xgb | XGBoost | 173 | 64 | 109 | 36.99% | 36.99% | 36.99% | 13.01 pp | -45 | 9 | -5.00 |
| BTC Daily | lstm | LSTM | 163 | 61 | 102 | 37.42% | 37.42% | 37.42% | 12.58 pp | -41 | 8 | -5.12 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 161 | 83 | 78 | 51.55% | 51.55% | 51.55% | 1.55 pp | 5 | 13 | 0.38 |
| BTC Market Hours | rf | RandomForest | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | transformer | Transformer | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 161 | 72 | 89 | 44.72% | 44.72% | 44.72% | 5.28 pp | -17 | 13 | -1.31 |
| BTC Market Hours | lstm | LSTM | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |
| BTC Market Hours | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 161 | 79 | 82 | 49.07% | 49.07% | 49.07% | 0.93 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 161 | 77 | 84 | 47.83% | 47.83% | 47.83% | 2.17 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 14 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 14 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 161 | 66 | 95 | 40.99% | 40.99% | 40.99% | 9.01 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 161 | 62 | 99 | 38.51% | 38.51% | 38.51% | 11.49 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 135 | 68 | 67 | 50.37% | 50.37% | 50.37% | 0.37 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 135 | 66 | 69 | 48.89% | 48.89% | 48.89% | 1.11 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 135 | 58 | 77 | 42.96% | 42.96% | 42.96% | 7.04 pp | -19 | 11 | -1.73 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 30 | 12 | 18 | 40.00% | 40.00% | 40.00% | 10.00 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 30 | 8 | 22 | 26.67% | 26.67% | 26.67% | 23.33 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
