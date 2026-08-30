# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T03:20:51.767606+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 156 | 96 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 175 | 84 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 175 | 84 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 84 | 48 | 36 | 57.14% | 57.14% | 57.14% | 7.14 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 44 | 40 | 52.38% | 52.38% | 52.38% | 2.38 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 8 | -0.25 |
| BTC Market Hours | rf | RandomForest | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 7 | -0.29 |
| BTC Hourly | nn | NN | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| BTC Daily | transformer | Transformer | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | nn | NN | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 8 | -1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 3 | -1.67 |
| BTC Market Hours | transformer | Transformer | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 7 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 84 | 34 | 50 | 40.48% | 40.48% | 40.48% | 9.52 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 84 | 32 | 52 | 38.10% | 38.10% | 38.10% | 11.90 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| BTC Hourly | rf | RandomForest | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 3 | -5.00 |
| BTC Daily | rf | RandomForest | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 61 | 19 | 42 | 31.15% | 31.15% | 31.15% | 18.85 pp | -23 | 3 | -7.67 |
| BTC Daily | xgb | XGBoost | 96 | 28 | 68 | 29.17% | 29.17% | 29.17% | 20.83 pp | -40 | 5 | -8.00 |

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
| BTC Daily | nn | NN | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 4 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 96 | 28 | 68 | 29.17% | 29.17% | 29.17% | 20.83 pp | -40 | 5 | -8.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 84 | 48 | 36 | 57.14% | 57.14% | 57.14% | 7.14 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | rf | RandomForest | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 44 | 40 | 52.38% | 52.38% | 52.38% | 2.38 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 84 | 41 | 43 | 48.81% | 48.81% | 48.81% | 1.19 pp | -2 | 8 | -0.25 |
| BTC Market Hours Daily | rf | RandomForest | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 84 | 36 | 48 | 42.86% | 42.86% | 42.86% | 7.14 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 84 | 34 | 50 | 40.48% | 40.48% | 40.48% | 9.52 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 84 | 32 | 52 | 38.10% | 38.10% | 38.10% | 11.90 pp | -20 | 8 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

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
