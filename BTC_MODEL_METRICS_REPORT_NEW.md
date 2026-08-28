# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T15:50:53.735464+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 33 | 69 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 129 | 69 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 14:00:00+00:00 | 125 | 57 | 68 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 14:00:00+00:00 | 125 | 57 | 68 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 1 | 43 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 16:00:00+00:00 | 44 | 1 | 43 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 57 | 31 | 26 | 54.39% | 54.39% | 54.39% | 4.39 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| BTC Hourly | nn | NN | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Hourly | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |
| BTC Hourly | lstm | LSTM | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 5 | -3.40 |
| BTC Daily | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Market Hours Daily | lstm | LSTM | 57 | 17 | 40 | 29.82% | 29.82% | 29.82% | 20.18 pp | -23 | 6 | -3.83 |
| BTC Daily | lstm | LSTM | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 69 | 23 | 46 | 33.33% | 33.33% | 33.33% | 16.67 pp | -23 | 4 | -5.75 |
| BTC Hourly | rf | RandomForest | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| BTC Hourly | nn | NN | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 2 | -1.50 |
| BTC Hourly | lstm | LSTM | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 69 | 23 | 46 | 33.33% | 33.33% | 33.33% | 16.67 pp | -23 | 4 | -5.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 57 | 31 | 26 | 54.39% | 54.39% | 54.39% | 4.39 pp | 5 | 5 | 1.00 |
| BTC Market Hours | rf | RandomForest | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| BTC Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Market Hours | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 5 | -3.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | lstm | LSTM | 57 | 17 | 40 | 29.82% | 29.82% | 29.82% | 20.18 pp | -23 | 6 | -3.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 26 | 18 | 59.09% | 59.09% | 59.09% | 9.09 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
