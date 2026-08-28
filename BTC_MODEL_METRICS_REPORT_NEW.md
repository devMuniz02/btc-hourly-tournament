# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T16:14:14.913903+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 33 | 69 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 129 | 69 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 126 | 57 | 69 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 126 | 57 | 69 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 0 | 43 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 0 | 43 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| BTC Hourly | nn | NN | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| BTC Market Hours | nn | NN | 57 | 31 | 26 | 54.39% | 54.39% | 54.39% | 4.39 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 2 | -0.50 |
| BTC Daily | nn | NN | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| BTC Hourly | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| BTC Market Hours | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 5 | -3.40 |
| BTC Daily | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Market Hours Daily | lstm | LSTM | 57 | 17 | 40 | 29.82% | 29.82% | 29.82% | 20.18 pp | -23 | 6 | -3.83 |
| BTC Daily | lstm | LSTM | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 69 | 23 | 46 | 33.33% | 33.33% | 33.33% | 16.67 pp | -23 | 4 | -5.75 |
| BTC Hourly | rf | RandomForest | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| BTC Hourly | nn | NN | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 2 | 1.50 |
| BTC Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 2 | -1.50 |
| BTC Hourly | rf | RandomForest | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 2 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 59 | 28 | 31 | 47.46% | 47.46% | 47.46% | 2.54 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 69 | 23 | 46 | 33.33% | 33.33% | 33.33% | 16.67 pp | -23 | 4 | -5.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 57 | 31 | 26 | 54.39% | 54.39% | 54.39% | 4.39 pp | 5 | 5 | 1.00 |
| BTC Market Hours | rf | RandomForest | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 57 | 26 | 31 | 45.61% | 45.61% | 45.61% | 4.39 pp | -5 | 5 | -1.00 |
| BTC Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Market Hours | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 5 | -3.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | lstm | LSTM | 57 | 17 | 40 | 29.82% | 29.82% | 29.82% | 20.18 pp | -23 | 6 | -3.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

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
