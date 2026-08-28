# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T12:28:12.402534+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 31 | 71 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 126 | 66 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 119 | 54 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 119 | 54 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| BTC Market Hours | nn | NN | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours | transformer | Transformer | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 3 | -1.33 |
| BTC Hourly | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | nn | NN | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 6 | -2.00 |
| BTC Hourly | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Market Hours | lstm | LSTM | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Market Hours Daily | lstm | LSTM | 54 | 16 | 38 | 29.63% | 29.63% | 29.63% | 20.37 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 66 | 22 | 44 | 33.33% | 33.33% | 33.33% | 16.67 pp | -22 | 4 | -5.50 |
| BTC Hourly | rf | RandomForest | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 2 | 1.50 |
| BTC Hourly | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 2 | -1.50 |
| BTC Hourly | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 56 | 26 | 30 | 46.43% | 46.43% | 46.43% | 3.57 pp | -4 | 3 | -1.33 |
| BTC Daily | rf | RandomForest | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 66 | 22 | 44 | 33.33% | 33.33% | 33.33% | 16.67 pp | -22 | 4 | -5.50 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | nn | NN | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 54 | 16 | 38 | 29.63% | 29.63% | 29.63% | 20.37 pp | -22 | 6 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

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
