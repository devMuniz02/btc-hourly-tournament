# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T01:07:50.479791+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 22 | 80 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 117 | 57 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 110 | 45 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 110 | 45 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 06:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 06:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 06:00:00+00:00 | 33 | 0 | 33 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 06:00:00+00:00 | 33 | 0 | 33 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| BTC Daily | transformer | Transformer | 47 | 25 | 22 | 53.19% | 53.19% | 53.19% | 3.19 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| BTC Market Hours | nn | NN | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| BTC Hourly | lstm | LSTM | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | rf | RandomForest | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | nn | NN | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| Consolidated Hourly | nn | NN | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 57 | 21 | 36 | 36.84% | 36.84% | 36.84% | 13.16 pp | -15 | 4 | -3.75 |
| BTC Hourly | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 1 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 45 | 12 | 33 | 26.67% | 26.67% | 26.67% | 23.33 pp | -21 | 5 | -4.20 |
| BTC Market Hours | lstm | LSTM | 45 | 14 | 31 | 31.11% | 31.11% | 31.11% | 18.89 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 3 | -5.00 |
| BTC Hourly | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 1 | 2.00 |
| BTC Hourly | lstm | LSTM | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 47 | 25 | 22 | 53.19% | 53.19% | 53.19% | 3.19 pp | 3 | 3 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 3 | -0.33 |
| BTC Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 3 | -1.67 |
| BTC Daily | xgb | XGBoost | 57 | 21 | 36 | 36.84% | 36.84% | 36.84% | 13.16 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 3 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 4 | 0.75 |
| BTC Market Hours | rf | RandomForest | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 45 | 18 | 27 | 40.00% | 40.00% | 40.00% | 10.00 pp | -9 | 4 | -2.25 |
| BTC Market Hours | lstm | LSTM | 45 | 14 | 31 | 31.11% | 31.11% | 31.11% | 18.89 pp | -17 | 4 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 45 | 12 | 33 | 26.67% | 26.67% | 26.67% | 23.33 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 4 | -3.25 |

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
