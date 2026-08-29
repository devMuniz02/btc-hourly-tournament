# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T15:34:41.093124+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 112 | 52 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 147 | 87 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 156 | 75 | 81 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 156 | 75 | 81 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 75 | 44 | 31 | 58.67% | 58.67% | 58.67% | 8.67 pp | 13 | 6 | 2.17 |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | transformer | Transformer | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 7 | -0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 77 | 37 | 40 | 48.05% | 48.05% | 48.05% | 1.95 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 77 | 37 | 40 | 48.05% | 48.05% | 48.05% | 1.95 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 77 | 36 | 41 | 46.75% | 46.75% | 46.75% | 3.25 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 7 | -1.29 |
| BTC Hourly | nn | NN | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | nn | NN | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 7 | -2.14 |
| BTC Market Hours | transformer | Transformer | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 6 | -2.50 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Hourly | xgb | XGBoost | 52 | 16 | 36 | 30.77% | 30.77% | 30.77% | 19.23 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 87 | 26 | 61 | 29.89% | 29.89% | 29.89% | 20.11 pp | -35 | 5 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 52 | 25 | 27 | 48.08% | 48.08% | 48.08% | 1.92 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 3 | -1.33 |
| BTC Hourly | lstm | LSTM | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 3 | -4.00 |
| BTC Hourly | xgb | XGBoost | 52 | 16 | 36 | 30.77% | 30.77% | 30.77% | 19.23 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 77 | 37 | 40 | 48.05% | 48.05% | 48.05% | 1.95 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 77 | 37 | 40 | 48.05% | 48.05% | 48.05% | 1.95 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 77 | 36 | 41 | 46.75% | 46.75% | 46.75% | 3.25 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Daily | xgb | XGBoost | 87 | 26 | 61 | 29.89% | 29.89% | 29.89% | 20.11 pp | -35 | 5 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 75 | 44 | 31 | 58.67% | 58.67% | 58.67% | 8.67 pp | 13 | 6 | 2.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 6 | -0.17 |
| BTC Market Hours | lstm | LSTM | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 6 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 7 | 0.43 |
| BTC Market Hours Daily | transformer | Transformer | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | rf | RandomForest | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | nn | NN | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 75 | 30 | 45 | 40.00% | 40.00% | 40.00% | 10.00 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 7 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
