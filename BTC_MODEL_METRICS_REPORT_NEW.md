# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T18:33:01.172810+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 133 | 73 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 168 | 108 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 17:00:00+00:00 | 193 | 96 | 97 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 17:00:00+00:00 | 193 | 96 | 97 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 76 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 76 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 0 | 76 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 0 | 76 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| BTC Market Hours | nn | NN | 96 | 52 | 44 | 54.17% | 54.17% | 54.17% | 4.17 pp | 8 | 8 | 1.00 |
| BTC Hourly | transformer | Transformer | 73 | 38 | 35 | 52.05% | 52.05% | 52.05% | 2.05 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 49 | 47 | 51.04% | 51.04% | 51.04% | 1.04 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | rf | RandomForest | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 8 | -0.25 |
| BTC Hourly | nn | NN | 73 | 36 | 37 | 49.32% | 49.32% | 49.32% | 0.68 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 98 | 46 | 52 | 46.94% | 46.94% | 46.94% | 3.06 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 9 | -1.56 |
| BTC Market Hours | lstm | LSTM | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 9 | -2.22 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 3 | -2.33 |
| BTC Daily | transformer | Transformer | 98 | 43 | 55 | 43.88% | 43.88% | 43.88% | 6.12 pp | -12 | 5 | -2.40 |
| BTC Market Hours | transformer | Transformer | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 9 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 98 | 38 | 60 | 38.78% | 38.78% | 38.78% | 11.22 pp | -22 | 5 | -4.40 |
| BTC Hourly | rf | RandomForest | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 3 | -5.00 |
| BTC Daily | lstm | LSTM | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 108 | 37 | 71 | 34.26% | 34.26% | 34.26% | 15.74 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 73 | 26 | 47 | 35.62% | 35.62% | 35.62% | 14.38 pp | -21 | 3 | -7.00 |
| BTC Hourly | xgb | XGBoost | 73 | 24 | 49 | 32.88% | 32.88% | 32.88% | 17.12 pp | -25 | 3 | -8.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 73 | 38 | 35 | 52.05% | 52.05% | 52.05% | 2.05 pp | 3 | 3 | 1.00 |
| BTC Hourly | nn | NN | 73 | 36 | 37 | 49.32% | 49.32% | 49.32% | 0.68 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 3 | -2.33 |
| BTC Hourly | rf | RandomForest | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 3 | -5.00 |
| BTC Hourly | lstm | LSTM | 73 | 26 | 47 | 35.62% | 35.62% | 35.62% | 14.38 pp | -21 | 3 | -7.00 |
| BTC Hourly | xgb | XGBoost | 73 | 24 | 49 | 32.88% | 32.88% | 32.88% | 17.12 pp | -25 | 3 | -8.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 98 | 46 | 52 | 46.94% | 46.94% | 46.94% | 3.06 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 98 | 43 | 55 | 43.88% | 43.88% | 43.88% | 6.12 pp | -12 | 5 | -2.40 |
| BTC Daily | rf | RandomForest | 98 | 38 | 60 | 38.78% | 38.78% | 38.78% | 11.22 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 108 | 37 | 71 | 34.26% | 34.26% | 34.26% | 15.74 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 96 | 52 | 44 | 54.17% | 54.17% | 54.17% | 4.17 pp | 8 | 8 | 1.00 |
| BTC Market Hours | rf | RandomForest | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 8 | -1.75 |
| BTC Market Hours | transformer | Transformer | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 49 | 47 | 51.04% | 51.04% | 51.04% | 1.04 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 9 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

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
