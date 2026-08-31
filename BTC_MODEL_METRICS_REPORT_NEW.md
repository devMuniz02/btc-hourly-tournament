# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T23:47:25.977783+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 153 | 93 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 189 | 129 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 22:00:00+00:00 | 232 | 117 | 115 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 22:00:00+00:00 | 232 | 117 | 115 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 96 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 117 | 64 | 53 | 54.70% | 54.70% | 54.70% | 4.70 pp | 11 | 9 | 1.22 |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | transformer | Transformer | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 117 | 57 | 60 | 48.72% | 48.72% | 48.72% | 1.28 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 6 | -0.50 |
| BTC Market Hours | rf | RandomForest | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 9 | -0.56 |
| BTC Hourly | nn | NN | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | transformer | Transformer | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Daily | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 6 | -1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 6 | -2.17 |
| BTC Market Hours | transformer | Transformer | 117 | 48 | 69 | 41.03% | 41.03% | 41.03% | 8.97 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 117 | 46 | 71 | 39.32% | 39.32% | 39.32% | 10.68 pp | -25 | 10 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 117 | 47 | 70 | 40.17% | 40.17% | 40.17% | 9.83 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 117 | 46 | 71 | 39.32% | 39.32% | 39.32% | 10.68 pp | -25 | 9 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 10 | -3.10 |
| BTC Daily | rf | RandomForest | 119 | 48 | 71 | 40.34% | 40.34% | 40.34% | 9.66 pp | -23 | 6 | -3.83 |
| BTC Hourly | rf | RandomForest | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 4 | -4.75 |
| BTC Daily | xgb | XGBoost | 129 | 47 | 82 | 36.43% | 36.43% | 36.43% | 13.57 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 119 | 41 | 78 | 34.45% | 34.45% | 34.45% | 15.55 pp | -37 | 6 | -6.17 |
| BTC Hourly | lstm | LSTM | 93 | 30 | 63 | 32.26% | 32.26% | 32.26% | 17.74 pp | -33 | 4 | -8.25 |
| BTC Hourly | xgb | XGBoost | 93 | 30 | 63 | 32.26% | 32.26% | 32.26% | 17.74 pp | -33 | 4 | -8.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 4 | -0.25 |
| BTC Hourly | nn | NN | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 4 | -4.75 |
| BTC Hourly | lstm | LSTM | 93 | 30 | 63 | 32.26% | 32.26% | 32.26% | 17.74 pp | -33 | 4 | -8.25 |
| BTC Hourly | xgb | XGBoost | 93 | 30 | 63 | 32.26% | 32.26% | 32.26% | 17.74 pp | -33 | 4 | -8.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 119 | 58 | 61 | 48.74% | 48.74% | 48.74% | 1.26 pp | -3 | 6 | -0.50 |
| BTC Daily | nn | NN | 119 | 55 | 64 | 46.22% | 46.22% | 46.22% | 3.78 pp | -9 | 6 | -1.50 |
| BTC Daily | transformer | Transformer | 119 | 53 | 66 | 44.54% | 44.54% | 44.54% | 5.46 pp | -13 | 6 | -2.17 |
| BTC Daily | rf | RandomForest | 119 | 48 | 71 | 40.34% | 40.34% | 40.34% | 9.66 pp | -23 | 6 | -3.83 |
| BTC Daily | xgb | XGBoost | 129 | 47 | 82 | 36.43% | 36.43% | 36.43% | 13.57 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 119 | 41 | 78 | 34.45% | 34.45% | 34.45% | 15.55 pp | -37 | 6 | -6.17 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 117 | 64 | 53 | 54.70% | 54.70% | 54.70% | 4.70 pp | 11 | 9 | 1.22 |
| BTC Market Hours | rf | RandomForest | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 9 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 9 | -1.00 |
| BTC Market Hours | transformer | Transformer | 117 | 48 | 69 | 41.03% | 41.03% | 41.03% | 8.97 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 117 | 47 | 70 | 40.17% | 40.17% | 40.17% | 9.83 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 117 | 46 | 71 | 39.32% | 39.32% | 39.32% | 10.68 pp | -25 | 9 | -2.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 117 | 57 | 60 | 48.72% | 48.72% | 48.72% | 1.28 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | transformer | Transformer | 117 | 52 | 65 | 44.44% | 44.44% | 44.44% | 5.56 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 117 | 46 | 71 | 39.32% | 39.32% | 39.32% | 10.68 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 10 | -3.10 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
