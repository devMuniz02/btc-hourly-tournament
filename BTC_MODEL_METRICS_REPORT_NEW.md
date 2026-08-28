# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T03:37:30.428643+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 23 | 79 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 119 | 59 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 112 | 47 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 112 | 47 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 0 | 35 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 0 | 35 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| BTC Market Hours | nn | NN | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 4 | 1.25 |
| BTC Daily | transformer | Transformer | 49 | 26 | 23 | 53.06% | 53.06% | 53.06% | 3.06 pp | 3 | 3 | 1.00 |
| BTC Hourly | lstm | LSTM | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 1 | 1.00 |
| BTC Hourly | nn | NN | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| BTC Market Hours | rf | RandomForest | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Daily | rf | RandomForest | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| BTC Market Hours Daily | nn | NN | 47 | 19 | 28 | 40.43% | 40.43% | 40.43% | 9.57 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 47 | 19 | 28 | 40.43% | 40.43% | 40.43% | 9.57 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 4 | -3.75 |
| BTC Market Hours Daily | lstm | LSTM | 47 | 13 | 34 | 27.66% | 27.66% | 27.66% | 22.34 pp | -21 | 5 | -4.20 |
| BTC Market Hours | lstm | LSTM | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |
| BTC Hourly | rf | RandomForest | 23 | 9 | 14 | 39.13% | 39.13% | 39.13% | 10.87 pp | -5 | 1 | -5.00 |
| BTC Daily | lstm | LSTM | 49 | 16 | 33 | 32.65% | 32.65% | 32.65% | 17.35 pp | -17 | 3 | -5.67 |
| BTC Hourly | xgb | XGBoost | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 1 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 1 | 3.00 |
| BTC Hourly | lstm | LSTM | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 1 | 1.00 |
| BTC Hourly | nn | NN | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 1 | 1.00 |
| BTC Hourly | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 23 | 9 | 14 | 39.13% | 39.13% | 39.13% | 10.87 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 1 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 49 | 26 | 23 | 53.06% | 53.06% | 53.06% | 3.06 pp | 3 | 3 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 3 | -0.33 |
| BTC Daily | rf | RandomForest | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 3 | -1.67 |
| BTC Daily | xgb | XGBoost | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 49 | 16 | 33 | 32.65% | 32.65% | 32.65% | 17.35 pp | -17 | 3 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 47 | 26 | 21 | 55.32% | 55.32% | 55.32% | 5.32 pp | 5 | 4 | 1.25 |
| BTC Market Hours | rf | RandomForest | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours | transformer | Transformer | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Market Hours | lstm | LSTM | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 24 | 23 | 51.06% | 51.06% | 51.06% | 1.06 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 47 | 19 | 28 | 40.43% | 40.43% | 40.43% | 9.57 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 47 | 19 | 28 | 40.43% | 40.43% | 40.43% | 9.57 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 47 | 13 | 34 | 27.66% | 27.66% | 27.66% | 22.34 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

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
