# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T08:11:39.292273+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 159 | 99 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 195 | 135 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 240 | 123 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 240 | 123 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T20:00:00+00:00 | 101 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T20:00:00+00:00 | 101 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T20:00:00+00:00 | 101 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T20:00:00+00:00 | 102 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 123 | 65 | 58 | 52.85% | 52.85% | 52.85% | 2.85 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | rf | RandomForest | 101 | 53 | 48 | 52.48% | 52.48% | 52.48% | 2.48 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 53 | 48 | 52.48% | 52.48% | 52.48% | 2.48 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 125 | 62 | 63 | 49.60% | 49.60% | 49.60% | 0.40 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 123 | 60 | 63 | 48.78% | 48.78% | 48.78% | 1.22 pp | -3 | 11 | -0.27 |
| BTC Market Hours | rf | RandomForest | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 6 | -1.83 |
| BTC Market Hours | transformer | Transformer | 123 | 52 | 71 | 42.28% | 42.28% | 42.28% | 7.72 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 123 | 48 | 75 | 39.02% | 39.02% | 39.02% | 10.98 pp | -27 | 11 | -2.45 |
| BTC Market Hours | xgb | XGBoost | 123 | 49 | 74 | 39.84% | 39.84% | 39.84% | 10.16 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 123 | 46 | 77 | 37.40% | 37.40% | 37.40% | 12.60 pp | -31 | 11 | -2.82 |
| BTC Market Hours | lstm | LSTM | 123 | 47 | 76 | 38.21% | 38.21% | 38.21% | 11.79 pp | -29 | 10 | -2.90 |
| BTC Daily | rf | RandomForest | 125 | 52 | 73 | 41.60% | 41.60% | 41.60% | 8.40 pp | -21 | 6 | -3.50 |
| BTC Hourly | rf | RandomForest | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 1 | -5.00 |
| BTC Daily | xgb | XGBoost | 135 | 49 | 86 | 36.30% | 36.30% | 36.30% | 13.70 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 125 | 43 | 82 | 34.40% | 34.40% | 34.40% | 15.60 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 99 | 32 | 67 | 32.32% | 32.32% | 32.32% | 17.68 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 99 | 31 | 68 | 31.31% | 31.31% | 31.31% | 18.69 pp | -37 | 5 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Hourly | rf | RandomForest | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 99 | 32 | 67 | 32.32% | 32.32% | 32.32% | 17.68 pp | -35 | 5 | -7.00 |
| BTC Hourly | lstm | LSTM | 99 | 31 | 68 | 31.31% | 31.31% | 31.31% | 18.69 pp | -37 | 5 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 125 | 62 | 63 | 49.60% | 49.60% | 49.60% | 0.40 pp | -1 | 6 | -0.17 |
| BTC Daily | nn | NN | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 6 | -1.83 |
| BTC Daily | rf | RandomForest | 125 | 52 | 73 | 41.60% | 41.60% | 41.60% | 8.40 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 135 | 49 | 86 | 36.30% | 36.30% | 36.30% | 13.70 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 125 | 43 | 82 | 34.40% | 34.40% | 34.40% | 15.60 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 123 | 65 | 58 | 52.85% | 52.85% | 52.85% | 2.85 pp | 7 | 10 | 0.70 |
| BTC Market Hours | rf | RandomForest | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| BTC Market Hours | transformer | Transformer | 123 | 52 | 71 | 42.28% | 42.28% | 42.28% | 7.72 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 123 | 49 | 74 | 39.84% | 39.84% | 39.84% | 10.16 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 123 | 47 | 76 | 38.21% | 38.21% | 38.21% | 11.79 pp | -29 | 10 | -2.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 123 | 60 | 63 | 48.78% | 48.78% | 48.78% | 1.22 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | rf | RandomForest | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 123 | 48 | 75 | 39.02% | 39.02% | 39.02% | 10.98 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 123 | 46 | 77 | 37.40% | 37.40% | 37.40% | 12.60 pp | -31 | 11 | -2.82 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 101 | 53 | 48 | 52.48% | 52.48% | 52.48% | 2.48 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 53 | 48 | 52.48% | 52.48% | 52.48% | 2.48 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
