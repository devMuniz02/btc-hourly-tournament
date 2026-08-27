# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T18:53:29.337936+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 17 | 85 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 113 | 53 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 17:00:00+00:00 | 99 | 41 | 58 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 17:00:00+00:00 | 99 | 41 | 58 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T02:00:00+00:00 | 29 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T02:00:00+00:00 | 29 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T02:00:00+00:00 | 29 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T02:00:00+00:00 | 30 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| BTC Market Hours | nn | NN | 41 | 23 | 18 | 56.10% | 56.10% | 56.10% | 6.10 pp | 5 | 4 | 1.25 |
| BTC Daily | transformer | Transformer | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Market Hours | rf | RandomForest | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 4 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 5 | 0.20 |
| BTC Market Hours | transformer | Transformer | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 1 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 3 | -2.33 |
| BTC Hourly | lstm | LSTM | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| BTC Hourly | transformer | Transformer | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |
| BTC Market Hours | lstm | LSTM | 41 | 13 | 28 | 31.71% | 31.71% | 31.71% | 18.29 pp | -15 | 4 | -3.75 |
| BTC Market Hours Daily | lstm | LSTM | 41 | 11 | 30 | 26.83% | 26.83% | 26.83% | 23.17 pp | -19 | 5 | -3.80 |
| BTC Hourly | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 1 | -5.00 |
| BTC Daily | lstm | LSTM | 43 | 14 | 29 | 32.56% | 32.56% | 32.56% | 17.44 pp | -15 | 3 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 1 | -1.00 |
| BTC Hourly | lstm | LSTM | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| BTC Hourly | transformer | Transformer | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 1 | -3.00 |
| BTC Hourly | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 1 | -5.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 3 | 1.00 |
| BTC Daily | nn | NN | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 3 | -2.33 |
| BTC Daily | xgb | XGBoost | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 43 | 14 | 29 | 32.56% | 32.56% | 32.56% | 17.44 pp | -15 | 3 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 41 | 23 | 18 | 56.10% | 56.10% | 56.10% | 6.10 pp | 5 | 4 | 1.25 |
| BTC Market Hours | rf | RandomForest | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 4 | 0.25 |
| BTC Market Hours | transformer | Transformer | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 41 | 13 | 28 | 31.71% | 31.71% | 31.71% | 18.29 pp | -15 | 4 | -3.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 21 | 20 | 51.22% | 51.22% | 51.22% | 1.22 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 41 | 11 | 30 | 26.83% | 26.83% | 26.83% | 23.17 pp | -19 | 5 | -3.80 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Hourly | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
