# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T16:01:34.432010+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 15 | 87 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 111 | 51 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 95 | 39 | 56 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 95 | 39 | 56 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 00:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 00:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 00:00:00+00:00 | 28 | 1 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 00:00:00+00:00 | 28 | 1 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 41 | 23 | 18 | 56.10% | 56.10% | 56.10% | 6.10 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| BTC Market Hours | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| BTC Market Hours | rf | RandomForest | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 3 | 0.33 |
| BTC Daily | nn | NN | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 4 | -0.75 |
| BTC Hourly | lstm | LSTM | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 3 | -1.67 |
| BTC Daily | xgb | XGBoost | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |
| BTC Market Hours Daily | lstm | LSTM | 39 | 11 | 28 | 28.21% | 28.21% | 28.21% | 21.79 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 3 | -4.33 |
| BTC Market Hours | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 3 | -4.33 |
| BTC Hourly | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | rf | RandomForest | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 1 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | rf | RandomForest | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 1 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 41 | 23 | 18 | 56.10% | 56.10% | 56.10% | 6.10 pp | 5 | 3 | 1.67 |
| BTC Daily | nn | NN | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 3 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 3 | -0.33 |
| BTC Daily | rf | RandomForest | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 3 | -1.67 |
| BTC Daily | xgb | XGBoost | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| BTC Daily | lstm | LSTM | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 3 | -4.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 3 | 1.00 |
| BTC Market Hours | rf | RandomForest | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 3 | -4.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 39 | 11 | 28 | 28.21% | 28.21% | 28.21% | 21.79 pp | -17 | 4 | -4.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
