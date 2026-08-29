# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T13:35:28.330406+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 110 | 50 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 146 | 86 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 12:00:00+00:00 | 153 | 74 | 79 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 12:00:00+00:00 | 153 | 74 | 79 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 59 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 74 | 43 | 31 | 58.11% | 58.11% | 58.11% | 8.11 pp | 12 | 6 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 7 | 0.29 |
| BTC Daily | transformer | Transformer | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 6 | -0.33 |
| BTC Market Hours | rf | RandomForest | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Hourly | nn | NN | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 76 | 36 | 40 | 47.37% | 47.37% | 47.37% | 2.63 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 7 | -2.29 |
| BTC Market Hours | xgb | XGBoost | 74 | 30 | 44 | 40.54% | 40.54% | 40.54% | 9.46 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| BTC Hourly | lstm | LSTM | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 3 | -3.33 |
| BTC Daily | rf | RandomForest | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 76 | 27 | 49 | 35.53% | 35.53% | 35.53% | 14.47 pp | -22 | 4 | -5.50 |
| BTC Daily | xgb | XGBoost | 86 | 27 | 59 | 31.40% | 31.40% | 31.40% | 18.60 pp | -32 | 5 | -6.40 |
| BTC Hourly | xgb | XGBoost | 50 | 15 | 35 | 30.00% | 30.00% | 30.00% | 20.00 pp | -20 | 3 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | nn | NN | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 3 | -0.67 |
| BTC Hourly | lstm | LSTM | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 3 | -3.33 |
| BTC Hourly | xgb | XGBoost | 50 | 15 | 35 | 30.00% | 30.00% | 30.00% | 20.00 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 76 | 36 | 40 | 47.37% | 47.37% | 47.37% | 2.63 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 76 | 27 | 49 | 35.53% | 35.53% | 35.53% | 14.47 pp | -22 | 4 | -5.50 |
| BTC Daily | xgb | XGBoost | 86 | 27 | 59 | 31.40% | 31.40% | 31.40% | 18.60 pp | -32 | 5 | -6.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 74 | 43 | 31 | 58.11% | 58.11% | 58.11% | 8.11 pp | 12 | 6 | 2.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 6 | -0.33 |
| BTC Market Hours | rf | RandomForest | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 6 | -0.33 |
| BTC Market Hours | lstm | LSTM | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 74 | 30 | 44 | 40.54% | 40.54% | 40.54% | 9.46 pp | -14 | 6 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 74 | 38 | 36 | 51.35% | 51.35% | 51.35% | 1.35 pp | 2 | 7 | 0.29 |
| BTC Market Hours Daily | transformer | Transformer | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | nn | NN | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | lstm | LSTM | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 7 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
