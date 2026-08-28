# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T11:30:56.929411+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 30 | 72 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 126 | 66 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 119 | 54 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 118 | 53 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 01:00:00+00:00 | 39 | 39 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 01:00:00+00:00 | 39 | 39 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 01:00:00+00:00 | 39 | 0 | 39 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 01:00:00+00:00 | 39 | 0 | 39 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 2 | 2.00 |
| BTC Market Hours | nn | NN | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 5 | 0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | nn | NN | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| BTC Market Hours | rf | RandomForest | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 5 | -0.80 |
| BTC Hourly | lstm | LSTM | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Hourly | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours | xgb | XGBoost | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| BTC Market Hours | lstm | LSTM | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 3 | -3.33 |
| BTC Market Hours Daily | lstm | LSTM | 53 | 16 | 37 | 30.19% | 30.19% | 30.19% | 19.81 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 66 | 23 | 43 | 34.85% | 34.85% | 34.85% | 15.15 pp | -20 | 4 | -5.00 |
| BTC Hourly | rf | RandomForest | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 2 | -5.00 |
| BTC Daily | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 3 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 2 | 2.00 |
| BTC Hourly | nn | NN | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | lstm | LSTM | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 30 | 10 | 20 | 33.33% | 33.33% | 33.33% | 16.67 pp | -10 | 2 | -5.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 56 | 28 | 28 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 3 | -1.33 |
| BTC Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 66 | 23 | 43 | 34.85% | 34.85% | 34.85% | 15.15 pp | -20 | 4 | -5.00 |
| BTC Daily | lstm | LSTM | 56 | 19 | 37 | 33.93% | 33.93% | 33.93% | 16.07 pp | -18 | 3 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 5 | 1.20 |
| BTC Market Hours | rf | RandomForest | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours | xgb | XGBoost | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| BTC Market Hours | lstm | LSTM | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 53 | 16 | 37 | 30.19% | 30.19% | 30.19% | 19.81 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

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
