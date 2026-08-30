# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T22:57:36.838844+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 136 | 76 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 172 | 112 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 201 | 100 | 101 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 201 | 100 | 101 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 20:00:00+00:00 | 81 | 81 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 20:00:00+00:00 | 81 | 81 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 20:00:00+00:00 | 81 | 1 | 80 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 20:00:00+00:00 | 81 | 1 | 80 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 100 | 55 | 45 | 55.00% | 55.00% | 55.00% | 5.00 pp | 10 | 8 | 1.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Hourly | transformer | Transformer | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 100 | 51 | 49 | 51.00% | 51.00% | 51.00% | 1.00 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| BTC Market Hours | rf | RandomForest | 100 | 49 | 51 | 49.00% | 49.00% | 49.00% | 1.00 pp | -2 | 8 | -0.25 |
| BTC Hourly | nn | NN | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 8 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | nn | NN | 100 | 44 | 56 | 44.00% | 44.00% | 44.00% | 6.00 pp | -12 | 9 | -1.33 |
| BTC Daily | nn | NN | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 5 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 100 | 42 | 58 | 42.00% | 42.00% | 42.00% | 8.00 pp | -16 | 8 | -2.00 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 9 | -2.22 |
| BTC Market Hours | transformer | Transformer | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 100 | 37 | 63 | 37.00% | 37.00% | 37.00% | 13.00 pp | -26 | 9 | -2.89 |
| BTC Market Hours | xgb | XGBoost | 100 | 38 | 62 | 38.00% | 38.00% | 38.00% | 12.00 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 102 | 37 | 65 | 36.27% | 36.27% | 36.27% | 13.73 pp | -28 | 5 | -5.60 |
| BTC Hourly | lstm | LSTM | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 112 | 38 | 74 | 33.93% | 33.93% | 33.93% | 16.07 pp | -36 | 6 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 4 | 1.00 |
| BTC Hourly | nn | NN | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 4 | -3.50 |
| BTC Hourly | lstm | LSTM | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 102 | 37 | 65 | 36.27% | 36.27% | 36.27% | 13.73 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 112 | 38 | 74 | 33.93% | 33.93% | 33.93% | 16.07 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 100 | 55 | 45 | 55.00% | 55.00% | 55.00% | 5.00 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 100 | 49 | 51 | 49.00% | 49.00% | 49.00% | 1.00 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 100 | 42 | 58 | 42.00% | 42.00% | 42.00% | 8.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | transformer | Transformer | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 100 | 38 | 62 | 38.00% | 38.00% | 38.00% | 12.00 pp | -24 | 8 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 100 | 51 | 49 | 51.00% | 51.00% | 51.00% | 1.00 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 100 | 44 | 56 | 44.00% | 44.00% | 44.00% | 6.00 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | lstm | LSTM | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 100 | 37 | 63 | 37.00% | 37.00% | 37.00% | 13.00 pp | -26 | 9 | -2.89 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | transformer | Transformer | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 36 | 45 | 44.44% | 44.44% | 44.44% | 5.56 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
