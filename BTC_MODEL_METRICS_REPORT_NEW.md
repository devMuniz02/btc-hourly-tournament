# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T15:51:34.580187+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 131 | 71 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 166 | 106 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 188 | 94 | 94 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 188 | 94 | 94 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 0 | 74 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 0 | 74 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 3 | 1.67 |
| BTC Market Hours | nn | NN | 94 | 51 | 43 | 54.26% | 54.26% | 54.26% | 4.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 48 | 46 | 51.06% | 51.06% | 51.06% | 1.06 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | rf | RandomForest | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Hourly | nn | NN | 71 | 34 | 37 | 47.89% | 47.89% | 47.89% | 2.11 pp | -3 | 3 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 45 | 51 | 46.88% | 46.88% | 46.88% | 3.12 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | nn | NN | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 9 | -1.56 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 9 | -2.22 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 96 | 42 | 54 | 43.75% | 43.75% | 43.75% | 6.25 pp | -12 | 5 | -2.40 |
| BTC Market Hours | transformer | Transformer | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 9 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 3 | -3.00 |
| BTC Daily | rf | RandomForest | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 5 | -4.40 |
| BTC Hourly | rf | RandomForest | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 3 | -5.00 |
| BTC Daily | lstm | LSTM | 96 | 35 | 61 | 36.46% | 36.46% | 36.46% | 13.54 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 71 | 22 | 49 | 30.99% | 30.99% | 30.99% | 19.01 pp | -27 | 3 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 3 | 1.67 |
| BTC Hourly | nn | NN | 71 | 34 | 37 | 47.89% | 47.89% | 47.89% | 2.11 pp | -3 | 3 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 3 | -5.00 |
| BTC Hourly | lstm | LSTM | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 71 | 22 | 49 | 30.99% | 30.99% | 30.99% | 19.01 pp | -27 | 3 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 45 | 51 | 46.88% | 46.88% | 46.88% | 3.12 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 96 | 42 | 54 | 43.75% | 43.75% | 43.75% | 6.25 pp | -12 | 5 | -2.40 |
| BTC Daily | rf | RandomForest | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 96 | 35 | 61 | 36.46% | 36.46% | 36.46% | 13.54 pp | -26 | 5 | -5.20 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 48 | 46 | 51.06% | 51.06% | 51.06% | 1.06 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 9 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

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
