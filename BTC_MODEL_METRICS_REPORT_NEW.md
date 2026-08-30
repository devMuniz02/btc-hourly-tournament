# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T04:38:38.729707+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 157 | 97 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 176 | 85 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 176 | 85 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 0 | 67 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 0 | 67 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 85 | 48 | 37 | 56.47% | 56.47% | 56.47% | 6.47 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| BTC Hourly | nn | NN | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | transformer | Transformer | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 8 | -1.12 |
| BTC Daily | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Market Hours | transformer | Transformer | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 8 | -1.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 62 | 23 | 39 | 37.10% | 37.10% | 37.10% | 12.90 pp | -16 | 3 | -5.33 |
| BTC Daily | lstm | LSTM | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 4 | -5.75 |
| BTC Daily | rf | RandomForest | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 4 | -5.75 |
| BTC Hourly | xgb | XGBoost | 62 | 20 | 42 | 32.26% | 32.26% | 32.26% | 17.74 pp | -22 | 3 | -7.33 |
| BTC Daily | xgb | XGBoost | 97 | 28 | 69 | 28.87% | 28.87% | 28.87% | 21.13 pp | -41 | 5 | -8.20 |

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
| BTC Daily | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 4 | -1.25 |
| BTC Daily | transformer | Transformer | 87 | 40 | 47 | 45.98% | 45.98% | 45.98% | 4.02 pp | -7 | 4 | -1.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 4 | -2.25 |
| BTC Daily | lstm | LSTM | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 4 | -5.75 |
| BTC Daily | rf | RandomForest | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 4 | -5.75 |
| BTC Daily | xgb | XGBoost | 97 | 28 | 69 | 28.87% | 28.87% | 28.87% | 21.13 pp | -41 | 5 | -8.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 85 | 48 | 37 | 56.47% | 56.47% | 56.47% | 6.47 pp | 11 | 7 | 1.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours | transformer | Transformer | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 85 | 33 | 52 | 38.82% | 38.82% | 38.82% | 11.18 pp | -19 | 7 | -2.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | transformer | Transformer | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 8 | -0.12 |
| BTC Market Hours Daily | rf | RandomForest | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 8 | -2.62 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

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
