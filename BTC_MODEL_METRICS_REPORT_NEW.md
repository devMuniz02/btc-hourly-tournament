# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T11:29:58.745197+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 12 | 90 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 107 | 47 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 87 | 35 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 87 | 35 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 09:00:00+00:00 | 25 | 25 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 09:00:00+00:00 | 25 | 25 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 09:00:00+00:00 | 25 | 0 | 25 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 09:00:00+00:00 | 25 | 0 | 25 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 1 | 2.00 |
| BTC Daily | transformer | Transformer | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 2 | 1.50 |
| BTC Market Hours | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| BTC Market Hours Daily | rf | RandomForest | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | xgb | XGBoost | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 2 | -1.50 |
| BTC Hourly | lstm | LSTM | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 2 | -3.50 |
| BTC Market Hours | lstm | LSTM | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Hourly | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |
| BTC Hourly | transformer | Transformer | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 2 | -5.50 |
| BTC Hourly | rf | RandomForest | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 1 | 2.00 |
| BTC Hourly | lstm | LSTM | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Hourly | transformer | Transformer | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 1 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 2 | -1.50 |
| BTC Daily | rf | RandomForest | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 2 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 3 | 1.00 |
| BTC Market Hours | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Market Hours | lstm | LSTM | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | transformer | Transformer | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| BTC Market Hours Daily | rf | RandomForest | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | xgb | XGBoost | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | nn | NN | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | lstm | LSTM | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |

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
