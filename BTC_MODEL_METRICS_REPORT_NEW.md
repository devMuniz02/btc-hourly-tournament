# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T02:17:55.321088+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 41 | 61 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 137 | 77 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 143 | 65 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 143 | 65 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 1 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 1 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 65 | 37 | 28 | 56.92% | 56.92% | 56.92% | 6.92 pp | 9 | 5 | 1.80 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | transformer | Transformer | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 6 | 0.50 |
| BTC Hourly | nn | NN | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | rf | RandomForest | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 67 | 33 | 34 | 49.25% | 49.25% | 49.25% | 0.75 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 6 | -1.17 |
| BTC Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Hourly | lstm | LSTM | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| BTC Market Hours | lstm | LSTM | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 6 | -3.17 |
| BTC Daily | rf | RandomForest | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 4 | -4.75 |
| BTC Daily | xgb | XGBoost | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 5 | -5.00 |
| BTC Hourly | rf | RandomForest | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 2 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 2 | -1.50 |
| BTC Hourly | rf | RandomForest | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 2 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 67 | 33 | 34 | 49.25% | 49.25% | 49.25% | 0.75 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 4 | -0.75 |
| BTC Daily | rf | RandomForest | 67 | 26 | 41 | 38.81% | 38.81% | 38.81% | 11.19 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 4 | -4.75 |
| BTC Daily | xgb | XGBoost | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 5 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 65 | 37 | 28 | 56.92% | 56.92% | 56.92% | 6.92 pp | 9 | 5 | 1.80 |
| BTC Market Hours | rf | RandomForest | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 65 | 26 | 39 | 40.00% | 40.00% | 40.00% | 10.00 pp | -13 | 5 | -2.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 6 | 0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 65 | 34 | 31 | 52.31% | 52.31% | 52.31% | 2.31 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | nn | NN | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | lstm | LSTM | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 6 | -3.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
