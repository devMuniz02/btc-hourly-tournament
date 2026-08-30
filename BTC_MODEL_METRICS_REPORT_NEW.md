# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T04:06:13.839540+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 121 | 61 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 157 | 97 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 176 | 85 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 176 | 85 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 18:00:00+00:00 | 68 | 68 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 18:00:00+00:00 | 68 | 68 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 18:00:00+00:00 | 68 | 1 | 67 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 18:00:00+00:00 | 68 | 1 | 67 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 85 | 48 | 37 | 56.47% | 56.47% | 56.47% | 6.47 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 68 | 36 | 32 | 52.94% | 52.94% | 52.94% | 2.94 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 68 | 36 | 32 | 52.94% | 52.94% | 52.94% | 2.94 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Hourly | nn | NN | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 7 | -1.14 |
| BTC Daily | transformer | Transformer | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | nn | NN | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 8 | -1.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | xgb | XGBoost | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 3 | -5.00 |
| BTC Daily | rf | RandomForest | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 4 | -5.25 |
| BTC Daily | lstm | LSTM | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 61 | 19 | 42 | 31.15% | 31.15% | 31.15% | 18.85 pp | -23 | 3 | -7.67 |
| BTC Daily | xgb | XGBoost | 97 | 29 | 68 | 29.90% | 29.90% | 29.90% | 20.10 pp | -39 | 5 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 3 | -1.67 |
| BTC Hourly | rf | RandomForest | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 3 | -5.00 |
| BTC Hourly | xgb | XGBoost | 61 | 19 | 42 | 31.15% | 31.15% | 31.15% | 18.85 pp | -23 | 3 | -7.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 4 | -1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Daily | rf | RandomForest | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 4 | -5.25 |
| BTC Daily | lstm | LSTM | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 4 | -6.25 |
| BTC Daily | xgb | XGBoost | 97 | 29 | 68 | 29.90% | 29.90% | 29.90% | 20.10 pp | -39 | 5 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 85 | 48 | 37 | 56.47% | 56.47% | 56.47% | 6.47 pp | 11 | 7 | 1.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours | transformer | Transformer | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | transformer | Transformer | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 8 | -0.12 |
| BTC Market Hours Daily | rf | RandomForest | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 8 | -2.62 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 68 | 36 | 32 | 52.94% | 52.94% | 52.94% | 2.94 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 68 | 36 | 32 | 52.94% | 52.94% | 52.94% | 2.94 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 68 | 30 | 38 | 44.12% | 44.12% | 44.12% | 5.88 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
