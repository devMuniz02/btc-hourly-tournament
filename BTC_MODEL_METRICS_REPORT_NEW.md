# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T14:35:48.557128+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 130 | 70 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 165 | 105 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 186 | 93 | 93 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 186 | 93 | 93 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 73 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 73 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 0 | 73 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 0 | 73 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 3 | 1.33 |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| Consolidated Hourly | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Hourly | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | nn | NN | 93 | 40 | 53 | 43.01% | 43.01% | 43.01% | 6.99 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 9 | -2.11 |
| Consolidated Hourly | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 5 | -2.20 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 9 | -2.56 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 3 | -2.67 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Hourly | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 70 | 22 | 48 | 31.43% | 31.43% | 31.43% | 18.57 pp | -26 | 3 | -8.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 3 | 1.33 |
| BTC Hourly | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 3 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 3 | -2.67 |
| BTC Hourly | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 3 | -4.67 |
| BTC Hourly | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 70 | 22 | 48 | 31.43% | 31.43% | 31.43% | 18.57 pp | -26 | 3 | -8.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | nn | NN | 93 | 40 | 53 | 43.01% | 43.01% | 43.01% | 6.99 pp | -13 | 9 | -1.44 |
| BTC Market Hours Daily | lstm | LSTM | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 9 | -2.56 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |

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
