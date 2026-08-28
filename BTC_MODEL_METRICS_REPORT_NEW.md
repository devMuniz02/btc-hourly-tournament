# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T15:03:04.041105+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 128 | 68 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 14:00:00+00:00 | 124 | 56 | 68 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 14:00:00+00:00 | 124 | 56 | 68 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 04:00:00+00:00 | 42 | 42 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 04:00:00+00:00 | 42 | 42 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 04:00:00+00:00 | 42 | 0 | 42 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 04:00:00+00:00 | 42 | 0 | 42 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 56 | 31 | 25 | 55.36% | 55.36% | 55.36% | 5.36 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| BTC Hourly | nn | NN | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours | rf | RandomForest | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Hourly | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 2 | -1.50 |
| BTC Market Hours | transformer | Transformer | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Hourly | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |
| BTC Hourly | lstm | LSTM | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 5 | -3.60 |
| BTC Market Hours Daily | lstm | LSTM | 56 | 17 | 39 | 30.36% | 30.36% | 30.36% | 19.64 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 68 | 22 | 46 | 32.35% | 32.35% | 32.35% | 17.65 pp | -24 | 4 | -6.00 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 3 | -1.33 |
| BTC Daily | rf | RandomForest | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 68 | 22 | 46 | 32.35% | 32.35% | 32.35% | 17.65 pp | -24 | 4 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 56 | 31 | 25 | 55.36% | 55.36% | 55.36% | 5.36 pp | 6 | 5 | 1.20 |
| BTC Market Hours | rf | RandomForest | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| BTC Market Hours | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 5 | -3.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 56 | 17 | 39 | 30.36% | 30.36% | 30.36% | 19.64 pp | -22 | 6 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |

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
