# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T23:50:10.538856+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 21 | 81 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 116 | 56 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 107 | 44 | 63 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 107 | 44 | 63 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 05:00:00+00:00 | 32 | 32 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 05:00:00+00:00 | 32 | 32 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 05:00:00+00:00 | 32 | 0 | 32 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 05:00:00+00:00 | 32 | 0 | 32 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| BTC Market Hours | nn | NN | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 4 | 1.00 |
| BTC Hourly | lstm | LSTM | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 1 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 1 | 1.00 |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | nn | NN | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Hourly | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 4 | -4.00 |
| BTC Market Hours | lstm | LSTM | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 44 | 11 | 33 | 25.00% | 25.00% | 25.00% | 25.00 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 3 | -4.67 |
| BTC Hourly | rf | RandomForest | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 1 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 1 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 1 | 1.00 |
| BTC Hourly | nn | NN | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 21 | 9 | 12 | 42.86% | 42.86% | 42.86% | 7.14 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 21 | 8 | 13 | 38.10% | 38.10% | 38.10% | 11.90 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 1 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 3 | -4.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 4 | 1.00 |
| BTC Market Hours | rf | RandomForest | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | xgb | XGBoost | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | nn | NN | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 44 | 11 | 33 | 25.00% | 25.00% | 25.00% | 25.00 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |

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
