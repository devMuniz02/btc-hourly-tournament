# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T10:50:16.793220+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 108 | 48 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 144 | 84 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 150 | 72 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 150 | 72 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 57 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 72 | 41 | 31 | 56.94% | 56.94% | 56.94% | 6.94 pp | 10 | 6 | 1.67 |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| BTC Daily | transformer | Transformer | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Hourly | nn | NN | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 2 | -1.00 |
| BTC Daily | nn | NN | 74 | 35 | 39 | 47.30% | 47.30% | 47.30% | 2.70 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | nn | NN | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| BTC Daily | rf | RandomForest | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 4 | -4.00 |
| BTC Hourly | lstm | LSTM | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 2 | -5.00 |
| BTC Daily | lstm | LSTM | 74 | 26 | 48 | 35.14% | 35.14% | 35.14% | 14.86 pp | -22 | 4 | -5.50 |
| BTC Hourly | rf | RandomForest | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 2 | -6.00 |
| BTC Daily | xgb | XGBoost | 84 | 26 | 58 | 30.95% | 30.95% | 30.95% | 19.05 pp | -32 | 5 | -6.40 |
| BTC Hourly | xgb | XGBoost | 48 | 15 | 33 | 31.25% | 31.25% | 31.25% | 18.75 pp | -18 | 2 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | nn | NN | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 2 | -5.00 |
| BTC Hourly | rf | RandomForest | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 48 | 15 | 33 | 31.25% | 31.25% | 31.25% | 18.75 pp | -18 | 2 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 74 | 36 | 38 | 48.65% | 48.65% | 48.65% | 1.35 pp | -2 | 4 | -0.50 |
| BTC Daily | nn | NN | 74 | 35 | 39 | 47.30% | 47.30% | 47.30% | 2.70 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 74 | 29 | 45 | 39.19% | 39.19% | 39.19% | 10.81 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 74 | 26 | 48 | 35.14% | 35.14% | 35.14% | 14.86 pp | -22 | 4 | -5.50 |
| BTC Daily | xgb | XGBoost | 84 | 26 | 58 | 30.95% | 30.95% | 30.95% | 19.05 pp | -32 | 5 | -6.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 72 | 41 | 31 | 56.94% | 56.94% | 56.94% | 6.94 pp | 10 | 6 | 1.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Market Hours | transformer | Transformer | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| BTC Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours Daily | nn | NN | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | xgb | XGBoost | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 7 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
