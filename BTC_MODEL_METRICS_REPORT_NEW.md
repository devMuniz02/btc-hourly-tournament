# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T17:40:23.309359+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 130 | 70 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 128 | 58 | 70 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 128 | 58 | 70 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| BTC Market Hours | nn | NN | 58 | 32 | 26 | 55.17% | 55.17% | 55.17% | 5.17 pp | 6 | 5 | 1.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Hourly | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | transformer | Transformer | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Daily | nn | NN | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 3 | -2.00 |
| BTC Market Hours | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 6 | -2.00 |
| BTC Hourly | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| BTC Market Hours | lstm | LSTM | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 5 | -3.20 |
| BTC Market Hours Daily | lstm | LSTM | 58 | 18 | 40 | 31.03% | 31.03% | 31.03% | 18.97 pp | -22 | 6 | -3.67 |
| BTC Daily | lstm | LSTM | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 3 | -4.67 |
| BTC Daily | rf | RandomForest | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 70 | 23 | 47 | 32.86% | 32.86% | 32.86% | 17.14 pp | -24 | 4 | -6.00 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 3 | -2.00 |
| BTC Daily | lstm | LSTM | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 3 | -4.67 |
| BTC Daily | rf | RandomForest | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 70 | 23 | 47 | 32.86% | 32.86% | 32.86% | 17.14 pp | -24 | 4 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 58 | 32 | 26 | 55.17% | 55.17% | 55.17% | 5.17 pp | 6 | 5 | 1.20 |
| BTC Market Hours | rf | RandomForest | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 58 | 27 | 31 | 46.55% | 46.55% | 46.55% | 3.45 pp | -4 | 5 | -0.80 |
| BTC Market Hours | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 5 | -3.20 |

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
