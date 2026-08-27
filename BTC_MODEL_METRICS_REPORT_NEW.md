# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T22:43:29.329038+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 20 | 82 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 116 | 56 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 21:00:00+00:00 | 106 | 44 | 62 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 21:00:00+00:00 | 105 | 43 | 62 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 0 | 31 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 04:00:00+00:00 | 31 | 0 | 31 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| BTC Market Hours | nn | NN | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | nn | NN | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Market Hours | lstm | LSTM | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 43 | 11 | 32 | 25.58% | 25.58% | 25.58% | 24.42 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 3 | -5.33 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 3 | -5.33 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | nn | NN | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 43 | 11 | 32 | 25.58% | 25.58% | 25.58% | 24.42 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

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
