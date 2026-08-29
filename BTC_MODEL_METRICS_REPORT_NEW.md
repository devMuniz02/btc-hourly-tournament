# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T00:05:06.873148+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 39 | 63 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 135 | 75 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 23:00:00+00:00 | 140 | 63 | 77 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 23:00:00+00:00 | 140 | 63 | 77 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 2 | 1.50 |
| BTC Hourly | transformer | Transformer | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 63 | 35 | 28 | 55.56% | 55.56% | 55.56% | 5.56 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Daily | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 4 | 0.25 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| BTC Market Hours | rf | RandomForest | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 6 | -1.17 |
| BTC Market Hours | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Hourly | lstm | LSTM | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours | lstm | LSTM | 63 | 24 | 39 | 38.10% | 38.10% | 38.10% | 11.90 pp | -15 | 5 | -3.00 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 6 | -3.17 |
| BTC Daily | rf | RandomForest | 65 | 25 | 40 | 38.46% | 38.46% | 38.46% | 11.54 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 75 | 25 | 50 | 33.33% | 33.33% | 33.33% | 16.67 pp | -25 | 5 | -5.00 |
| BTC Hourly | rf | RandomForest | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 2 | 1.50 |
| BTC Hourly | transformer | Transformer | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 2 | 1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 2 | -1.50 |
| BTC Hourly | rf | RandomForest | 39 | 14 | 25 | 35.90% | 35.90% | 35.90% | 14.10 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 4 | 0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 65 | 32 | 33 | 49.23% | 49.23% | 49.23% | 0.77 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 4 | -0.75 |
| BTC Daily | rf | RandomForest | 65 | 25 | 40 | 38.46% | 38.46% | 38.46% | 11.54 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 75 | 25 | 50 | 33.33% | 33.33% | 33.33% | 16.67 pp | -25 | 5 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 63 | 35 | 28 | 55.56% | 55.56% | 55.56% | 5.56 pp | 7 | 5 | 1.40 |
| BTC Market Hours | rf | RandomForest | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 63 | 24 | 39 | 38.10% | 38.10% | 38.10% | 11.90 pp | -15 | 5 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 63 | 33 | 30 | 52.38% | 52.38% | 52.38% | 2.38 pp | 3 | 6 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 6 | -1.17 |
| BTC Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 6 | -3.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

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
