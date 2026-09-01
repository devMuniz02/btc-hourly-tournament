# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T02:23:16.750919+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 155 | 95 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 191 | 131 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 236 | 119 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 235 | 118 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 119 | 64 | 55 | 53.78% | 53.78% | 53.78% | 3.78 pp | 9 | 10 | 0.90 |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 121 | 60 | 61 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 118 | 58 | 60 | 49.15% | 49.15% | 49.15% | 0.85 pp | -2 | 10 | -0.20 |
| BTC Market Hours | rf | RandomForest | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 119 | 56 | 63 | 47.06% | 47.06% | 47.06% | 2.94 pp | -7 | 10 | -0.70 |
| BTC Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | transformer | Transformer | 118 | 52 | 66 | 44.07% | 44.07% | 44.07% | 5.93 pp | -14 | 10 | -1.40 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 6 | -1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 4 | -1.75 |
| BTC Daily | transformer | Transformer | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 6 | -1.83 |
| BTC Market Hours | transformer | Transformer | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 10 | -1.90 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 119 | 48 | 71 | 40.34% | 40.34% | 40.34% | 9.66 pp | -23 | 10 | -2.30 |
| BTC Market Hours | lstm | LSTM | 119 | 47 | 72 | 39.50% | 39.50% | 39.50% | 10.50 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 118 | 46 | 72 | 38.98% | 38.98% | 38.98% | 11.02 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 118 | 44 | 74 | 37.29% | 37.29% | 37.29% | 12.71 pp | -30 | 10 | -3.00 |
| BTC Daily | rf | RandomForest | 121 | 50 | 71 | 41.32% | 41.32% | 41.32% | 8.68 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 95 | 38 | 57 | 40.00% | 40.00% | 40.00% | 10.00 pp | -19 | 4 | -4.75 |
| BTC Daily | xgb | XGBoost | 131 | 48 | 83 | 36.64% | 36.64% | 36.64% | 13.36 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 121 | 41 | 80 | 33.88% | 33.88% | 33.88% | 16.12 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 95 | 31 | 64 | 32.63% | 32.63% | 32.63% | 17.37 pp | -33 | 4 | -8.25 |
| BTC Hourly | lstm | LSTM | 95 | 30 | 65 | 31.58% | 31.58% | 31.58% | 18.42 pp | -35 | 4 | -8.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 95 | 38 | 57 | 40.00% | 40.00% | 40.00% | 10.00 pp | -19 | 4 | -4.75 |
| BTC Hourly | xgb | XGBoost | 95 | 31 | 64 | 32.63% | 32.63% | 32.63% | 17.37 pp | -33 | 4 | -8.25 |
| BTC Hourly | lstm | LSTM | 95 | 30 | 65 | 31.58% | 31.58% | 31.58% | 18.42 pp | -35 | 4 | -8.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 121 | 60 | 61 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 6 | -0.17 |
| BTC Daily | nn | NN | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 6 | -1.83 |
| BTC Daily | rf | RandomForest | 121 | 50 | 71 | 41.32% | 41.32% | 41.32% | 8.68 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 131 | 48 | 83 | 36.64% | 36.64% | 36.64% | 13.36 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 121 | 41 | 80 | 33.88% | 33.88% | 33.88% | 16.12 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 119 | 64 | 55 | 53.78% | 53.78% | 53.78% | 3.78 pp | 9 | 10 | 0.90 |
| BTC Market Hours | rf | RandomForest | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 119 | 56 | 63 | 47.06% | 47.06% | 47.06% | 2.94 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 119 | 50 | 69 | 42.02% | 42.02% | 42.02% | 7.98 pp | -19 | 10 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 119 | 48 | 71 | 40.34% | 40.34% | 40.34% | 9.66 pp | -23 | 10 | -2.30 |
| BTC Market Hours | lstm | LSTM | 119 | 47 | 72 | 39.50% | 39.50% | 39.50% | 10.50 pp | -25 | 10 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 118 | 58 | 60 | 49.15% | 49.15% | 49.15% | 0.85 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | nn | NN | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | transformer | Transformer | 118 | 52 | 66 | 44.07% | 44.07% | 44.07% | 5.93 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 118 | 46 | 72 | 38.98% | 38.98% | 38.98% | 11.02 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 118 | 44 | 74 | 37.29% | 37.29% | 37.29% | 12.71 pp | -30 | 10 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
