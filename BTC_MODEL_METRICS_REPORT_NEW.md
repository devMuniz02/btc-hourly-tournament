# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T22:57:48.346678+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 134 | 74 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 21:00:00+00:00 | 137 | 62 | 75 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 21:00:00+00:00 | 137 | 62 | 75 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 62 | 35 | 27 | 56.45% | 56.45% | 56.45% | 6.45 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| BTC Hourly | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 6 | 0.33 |
| BTC Daily | transformer | Transformer | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | rf | RandomForest | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 6 | -1.00 |
| BTC Market Hours | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 6 | -2.00 |
| BTC Hourly | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| BTC Market Hours Daily | lstm | LSTM | 62 | 21 | 41 | 33.87% | 33.87% | 33.87% | 16.13 pp | -20 | 6 | -3.33 |
| BTC Daily | lstm | LSTM | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 4 | -4.00 |
| BTC Daily | rf | RandomForest | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 4 | -4.00 |
| BTC Daily | xgb | XGBoost | 74 | 24 | 50 | 32.43% | 32.43% | 32.43% | 17.57 pp | -26 | 5 | -5.20 |
| BTC Hourly | rf | RandomForest | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 2 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | transformer | Transformer | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 2 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 64 | 32 | 32 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 4 | -1.00 |
| BTC Daily | lstm | LSTM | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 4 | -4.00 |
| BTC Daily | rf | RandomForest | 64 | 24 | 40 | 37.50% | 37.50% | 37.50% | 12.50 pp | -16 | 4 | -4.00 |
| BTC Daily | xgb | XGBoost | 74 | 24 | 50 | 32.43% | 32.43% | 32.43% | 17.57 pp | -26 | 5 | -5.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 62 | 35 | 27 | 56.45% | 56.45% | 56.45% | 6.45 pp | 8 | 5 | 1.60 |
| BTC Market Hours | rf | RandomForest | 62 | 31 | 31 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 6 | -1.00 |
| BTC Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 62 | 21 | 41 | 33.87% | 33.87% | 33.87% | 16.13 pp | -20 | 6 | -3.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

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
