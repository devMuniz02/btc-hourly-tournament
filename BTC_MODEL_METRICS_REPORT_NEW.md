# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T23:55:13.812315+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 137 | 77 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 173 | 113 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 203 | 101 | 102 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 203 | 101 | 102 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T10:00:00+00:00 | 81 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T10:00:00+00:00 | 81 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T10:00:00+00:00 | 81 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T10:00:00+00:00 | 82 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 8 | 1.12 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| BTC Hourly | transformer | Transformer | 77 | 40 | 37 | 51.95% | 51.95% | 51.95% | 1.95 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| BTC Hourly | nn | NN | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 4 | -1.75 |
| BTC Daily | nn | NN | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 101 | 42 | 59 | 41.58% | 41.58% | 41.58% | 8.42 pp | -17 | 8 | -2.12 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 9 | -2.33 |
| BTC Market Hours | transformer | Transformer | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | xgb | XGBoost | 101 | 37 | 64 | 36.63% | 36.63% | 36.63% | 13.37 pp | -27 | 9 | -3.00 |
| BTC Market Hours | xgb | XGBoost | 101 | 38 | 63 | 37.62% | 37.62% | 37.62% | 12.38 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 5 | -5.40 |
| BTC Hourly | xgb | XGBoost | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Daily | xgb | XGBoost | 113 | 39 | 74 | 34.51% | 34.51% | 34.51% | 15.49 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 4 | -6.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 77 | 40 | 37 | 51.95% | 51.95% | 51.95% | 1.95 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 4 | -3.25 |
| BTC Hourly | xgb | XGBoost | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Hourly | lstm | LSTM | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 4 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 113 | 39 | 74 | 34.51% | 34.51% | 34.51% | 15.49 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 8 | -0.62 |
| BTC Market Hours | lstm | LSTM | 101 | 42 | 59 | 41.58% | 41.58% | 41.58% | 8.42 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 101 | 38 | 63 | 37.62% | 37.62% | 37.62% | 12.38 pp | -25 | 8 | -3.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 101 | 51 | 50 | 50.50% | 50.50% | 50.50% | 0.50 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | rf | RandomForest | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Market Hours Daily | lstm | LSTM | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 101 | 37 | 64 | 36.63% | 36.63% | 36.63% | 13.37 pp | -27 | 9 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

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
