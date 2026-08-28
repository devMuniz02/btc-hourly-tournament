# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T17:50:30.224273+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 35 | 67 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 131 | 71 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 129 | 59 | 70 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 128 | 58 | 70 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 59 | 33 | 26 | 55.93% | 55.93% | 55.93% | 5.93 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Hourly | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 3 | 0.33 |
| BTC Market Hours | rf | RandomForest | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | transformer | Transformer | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 6 | -1.33 |
| BTC Daily | nn | NN | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 3 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 6 | -2.00 |
| BTC Hourly | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| BTC Market Hours | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 5 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 58 | 18 | 40 | 31.03% | 31.03% | 31.03% | 18.97 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 3 | -5.00 |
| BTC Daily | xgb | XGBoost | 71 | 24 | 47 | 33.80% | 33.80% | 33.80% | 16.20 pp | -23 | 4 | -5.75 |
| BTC Hourly | rf | RandomForest | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 2 | -7.50 |
| BTC Hourly | xgb | XGBoost | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 2 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Hourly | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 2 | -7.50 |
| BTC Hourly | xgb | XGBoost | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 2 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 3 | 0.33 |
| BTC Daily | nn | NN | 61 | 28 | 33 | 45.90% | 45.90% | 45.90% | 4.10 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 61 | 23 | 38 | 37.70% | 37.70% | 37.70% | 12.30 pp | -15 | 3 | -5.00 |
| BTC Daily | xgb | XGBoost | 71 | 24 | 47 | 33.80% | 33.80% | 33.80% | 16.20 pp | -23 | 4 | -5.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 59 | 33 | 26 | 55.93% | 55.93% | 55.93% | 5.93 pp | 7 | 5 | 1.40 |
| BTC Market Hours | rf | RandomForest | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 5 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 58 | 18 | 40 | 31.03% | 31.03% | 31.03% | 18.97 pp | -22 | 6 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |

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
