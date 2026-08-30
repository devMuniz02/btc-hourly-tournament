# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T14:45:35.026906+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 166 | 106 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 187 | 94 | 93 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 13:00:00+00:00 | 186 | 93 | 93 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 73 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 73 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 0 | 73 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 02:00:00+00:00 | 73 | 0 | 73 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 3 | 1.33 |
| BTC Market Hours | nn | NN | 94 | 51 | 43 | 54.26% | 54.26% | 54.26% | 4.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 73 | 40 | 33 | 54.79% | 54.79% | 54.79% | 4.79 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 73 | 39 | 34 | 53.42% | 53.42% | 53.42% | 3.42 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 73 | 37 | 36 | 50.68% | 50.68% | 50.68% | 0.68 pp | 1 | 8 | 0.12 |
| BTC Market Hours | rf | RandomForest | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 8 | -0.50 |
| BTC Hourly | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 9 | -0.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 73 | 32 | 41 | 43.84% | 43.84% | 43.84% | 6.16 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | nn | NN | 93 | 40 | 53 | 43.01% | 43.01% | 43.01% | 6.99 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 96 | 43 | 53 | 44.79% | 44.79% | 44.79% | 5.21 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 9 | -2.11 |
| Consolidated Hourly | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 9 | -2.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 3 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 5 | -4.00 |
| BTC Hourly | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 96 | 34 | 62 | 35.42% | 35.42% | 35.42% | 14.58 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 6 | -5.67 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 96 | 43 | 53 | 44.79% | 44.79% | 44.79% | 5.21 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 96 | 34 | 62 | 35.42% | 35.42% | 35.42% | 14.58 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 94 | 51 | 43 | 54.26% | 54.26% | 54.26% | 4.26 pp | 8 | 8 | 1.00 |
| BTC Market Hours | rf | RandomForest | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours | transformer | Transformer | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |

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
