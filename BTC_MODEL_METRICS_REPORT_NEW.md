# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T20:34:33.255979+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 115 | 55 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 151 | 91 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 19:00:00+00:00 | 165 | 79 | 86 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 19:00:00+00:00 | 165 | 79 | 86 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 01:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 01:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 01:00:00+00:00 | 61 | 0 | 61 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 01:00:00+00:00 | 61 | 0 | 61 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 79 | 46 | 33 | 58.23% | 58.23% | 58.23% | 8.23 pp | 13 | 7 | 1.86 |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 7 | 1.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 42 | 37 | 53.16% | 53.16% | 53.16% | 3.16 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| BTC Daily | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Hourly | nn | NN | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 81 | 39 | 42 | 48.15% | 48.15% | 48.15% | 1.85 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 7 | -1.00 |
| BTC Daily | nn | NN | 81 | 38 | 43 | 46.91% | 46.91% | 46.91% | 3.09 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | nn | NN | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| BTC Market Hours | transformer | Transformer | 79 | 34 | 45 | 43.04% | 43.04% | 43.04% | 6.96 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 79 | 32 | 47 | 40.51% | 40.51% | 40.51% | 9.49 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 79 | 32 | 47 | 40.51% | 40.51% | 40.51% | 9.49 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 81 | 28 | 53 | 34.57% | 34.57% | 34.57% | 15.43 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 55 | 18 | 37 | 32.73% | 32.73% | 32.73% | 17.27 pp | -19 | 3 | -6.33 |
| BTC Daily | xgb | XGBoost | 91 | 28 | 63 | 30.77% | 30.77% | 30.77% | 19.23 pp | -35 | 5 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Hourly | nn | NN | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Hourly | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 3 | -3.67 |
| BTC Hourly | xgb | XGBoost | 55 | 18 | 37 | 32.73% | 32.73% | 32.73% | 17.27 pp | -19 | 3 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 81 | 40 | 41 | 49.38% | 49.38% | 49.38% | 0.62 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 81 | 39 | 42 | 48.15% | 48.15% | 48.15% | 1.85 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 81 | 38 | 43 | 46.91% | 46.91% | 46.91% | 3.09 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 81 | 28 | 53 | 34.57% | 34.57% | 34.57% | 15.43 pp | -25 | 4 | -6.25 |
| BTC Daily | xgb | XGBoost | 91 | 28 | 63 | 30.77% | 30.77% | 30.77% | 19.23 pp | -35 | 5 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 79 | 46 | 33 | 58.23% | 58.23% | 58.23% | 8.23 pp | 13 | 7 | 1.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Market Hours | transformer | Transformer | 79 | 34 | 45 | 43.04% | 43.04% | 43.04% | 6.96 pp | -11 | 7 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 79 | 32 | 47 | 40.51% | 40.51% | 40.51% | 9.49 pp | -15 | 7 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 42 | 37 | 53.16% | 53.16% | 53.16% | 3.16 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | nn | NN | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | lstm | LSTM | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 79 | 32 | 47 | 40.51% | 40.51% | 40.51% | 9.49 pp | -15 | 7 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | nn | NN | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 21 | 40 | 34.43% | 34.43% | 34.43% | 15.57 pp | -19 | 7 | -2.71 |

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
