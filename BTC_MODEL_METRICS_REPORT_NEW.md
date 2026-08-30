# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T05:23:18.033131+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 122 | 62 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 158 | 98 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 177 | 86 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 177 | 86 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 19:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 19:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 19:00:00+00:00 | 69 | 1 | 68 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 19:00:00+00:00 | 69 | 1 | 68 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 86 | 49 | 37 | 56.98% | 56.98% | 56.98% | 6.98 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 38 | 31 | 55.07% | 55.07% | 55.07% | 5.07 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 38 | 31 | 55.07% | 55.07% | 55.07% | 5.07 pp | 7 | 7 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 86 | 43 | 43 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Hourly | nn | NN | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 8 | -0.25 |
| BTC Market Hours | rf | RandomForest | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 7 | -0.29 |
| BTC Daily | nn | NN | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Daily | transformer | Transformer | 88 | 41 | 47 | 46.59% | 46.59% | 46.59% | 3.41 pp | -6 | 4 | -1.50 |
| BTC Market Hours Daily | nn | NN | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 8 | -1.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 88 | 40 | 48 | 45.45% | 45.45% | 45.45% | 4.55 pp | -8 | 4 | -2.00 |
| BTC Market Hours | transformer | Transformer | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | xgb | XGBoost | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 8 | -2.75 |
| BTC Market Hours | xgb | XGBoost | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| BTC Hourly | rf | RandomForest | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 62 | 23 | 39 | 37.10% | 37.10% | 37.10% | 12.90 pp | -16 | 3 | -5.33 |
| BTC Daily | rf | RandomForest | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 62 | 20 | 42 | 32.26% | 32.26% | 32.26% | 17.74 pp | -22 | 3 | -7.33 |
| BTC Daily | xgb | XGBoost | 98 | 29 | 69 | 29.59% | 29.59% | 29.59% | 20.41 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 62 | 23 | 39 | 37.10% | 37.10% | 37.10% | 12.90 pp | -16 | 3 | -5.33 |
| BTC Hourly | xgb | XGBoost | 62 | 20 | 42 | 32.26% | 32.26% | 32.26% | 17.74 pp | -22 | 3 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 88 | 41 | 47 | 46.59% | 46.59% | 46.59% | 3.41 pp | -6 | 4 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 88 | 40 | 48 | 45.45% | 45.45% | 45.45% | 4.55 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 98 | 29 | 69 | 29.59% | 29.59% | 29.59% | 20.41 pp | -40 | 5 | -8.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 86 | 49 | 37 | 56.98% | 56.98% | 56.98% | 6.98 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 86 | 43 | 43 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | rf | RandomForest | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 8 | -0.25 |
| BTC Market Hours Daily | rf | RandomForest | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 8 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 38 | 31 | 55.07% | 55.07% | 55.07% | 5.07 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 38 | 31 | 55.07% | 55.07% | 55.07% | 5.07 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
