# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T16:49:13.032186+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 16 | 86 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 111 | 51 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 95 | 39 | 56 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 95 | 39 | 56 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| BTC Market Hours | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 3 | 1.00 |
| BTC Daily | transformer | Transformer | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Market Hours | rf | RandomForest | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Hourly | lstm | LSTM | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 41 | 17 | 24 | 41.46% | 41.46% | 41.46% | 8.54 pp | -7 | 3 | -2.33 |
| BTC Daily | xgb | XGBoost | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 4 | -3.75 |
| BTC Hourly | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 39 | 11 | 28 | 28.21% | 28.21% | 28.21% | 21.79 pp | -17 | 4 | -4.25 |
| BTC Daily | lstm | LSTM | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 3 | -4.33 |
| BTC Market Hours | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 3 | -4.33 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| BTC Hourly | xgb | XGBoost | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | lstm | LSTM | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 1 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 3 | 1.00 |
| BTC Daily | nn | NN | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 41 | 17 | 24 | 41.46% | 41.46% | 41.46% | 8.54 pp | -7 | 3 | -2.33 |
| BTC Daily | xgb | XGBoost | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 3 | -4.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 3 | 1.00 |
| BTC Market Hours | rf | RandomForest | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 3 | -1.00 |
| BTC Market Hours | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 3 | -4.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 39 | 21 | 18 | 53.85% | 53.85% | 53.85% | 3.85 pp | 3 | 4 | 0.75 |
| BTC Market Hours Daily | transformer | Transformer | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | xgb | XGBoost | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 39 | 11 | 28 | 28.21% | 28.21% | 28.21% | 21.79 pp | -17 | 4 | -4.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

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
