# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T03:37:05.944720+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 6 | 96 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 101 | 41 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 81 | 29 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 81 | 29 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 0 | 19 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 0 | 19 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| BTC Market Hours | nn | NN | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 3 | 1.67 |
| BTC Market Hours | rf | RandomForest | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 3 | 1.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| BTC Market Hours | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 4 | 0.25 |
| BTC Hourly | nn | NN | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| BTC Hourly | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| BTC Daily | rf | RandomForest | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 4 | -2.75 |
| BTC Daily | lstm | LSTM | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 3 | -3.67 |
| BTC Market Hours | lstm | LSTM | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 2 | 2.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 31 | 16 | 15 | 51.61% | 51.61% | 51.61% | 1.61 pp | 1 | 2 | 0.50 |
| BTC Daily | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Daily | rf | RandomForest | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Daily | lstm | LSTM | 31 | 12 | 19 | 38.71% | 38.71% | 38.71% | 11.29 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 3 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 3 | 1.67 |
| BTC Market Hours | rf | RandomForest | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 3 | 1.67 |
| BTC Market Hours | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 3 | -0.33 |
| BTC Market Hours | lstm | LSTM | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 4 | 1.25 |
| BTC Market Hours Daily | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | transformer | Transformer | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 4 | 0.25 |
| BTC Market Hours Daily | nn | NN | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | lstm | LSTM | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 4 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

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
