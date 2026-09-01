# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T01:47:45.819106+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 154 | 94 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 190 | 130 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 235 | 118 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 235 | 118 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 98 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 7 | 2 | 77.78% | 77.78% | 77.78% | 27.78 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 118 | 64 | 54 | 54.24% | 54.24% | 54.24% | 4.24 pp | 10 | 10 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 118 | 58 | 60 | 49.15% | 49.15% | 49.15% | 0.85 pp | -2 | 10 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 6 | -0.33 |
| BTC Market Hours | rf | RandomForest | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 10 | -0.80 |
| BTC Hourly | nn | NN | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | nn | NN | 118 | 54 | 64 | 45.76% | 45.76% | 45.76% | 4.24 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | transformer | Transformer | 118 | 52 | 66 | 44.07% | 44.07% | 44.07% | 5.93 pp | -14 | 10 | -1.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 4 | -1.50 |
| BTC Daily | nn | NN | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 6 | -2.00 |
| BTC Market Hours | transformer | Transformer | 118 | 49 | 69 | 41.53% | 41.53% | 41.53% | 8.47 pp | -20 | 10 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 118 | 47 | 71 | 39.83% | 39.83% | 39.83% | 10.17 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 118 | 46 | 72 | 38.98% | 38.98% | 38.98% | 11.02 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | xgb | XGBoost | 118 | 46 | 72 | 38.98% | 38.98% | 38.98% | 11.02 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 118 | 44 | 74 | 37.29% | 37.29% | 37.29% | 12.71 pp | -30 | 10 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 120 | 49 | 71 | 40.83% | 40.83% | 40.83% | 9.17 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 94 | 38 | 56 | 40.43% | 40.43% | 40.43% | 9.57 pp | -18 | 4 | -4.50 |
| BTC Daily | xgb | XGBoost | 130 | 47 | 83 | 36.15% | 36.15% | 36.15% | 13.85 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 120 | 41 | 79 | 34.17% | 34.17% | 34.17% | 15.83 pp | -38 | 6 | -6.33 |
| BTC Hourly | xgb | XGBoost | 94 | 31 | 63 | 32.98% | 32.98% | 32.98% | 17.02 pp | -32 | 4 | -8.00 |
| BTC Hourly | lstm | LSTM | 94 | 30 | 64 | 31.91% | 31.91% | 31.91% | 18.09 pp | -34 | 4 | -8.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 4 | -0.50 |
| BTC Hourly | nn | NN | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 94 | 38 | 56 | 40.43% | 40.43% | 40.43% | 9.57 pp | -18 | 4 | -4.50 |
| BTC Hourly | xgb | XGBoost | 94 | 31 | 63 | 32.98% | 32.98% | 32.98% | 17.02 pp | -32 | 4 | -8.00 |
| BTC Hourly | lstm | LSTM | 94 | 30 | 64 | 31.91% | 31.91% | 31.91% | 18.09 pp | -34 | 4 | -8.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 120 | 59 | 61 | 49.17% | 49.17% | 49.17% | 0.83 pp | -2 | 6 | -0.33 |
| BTC Daily | nn | NN | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 120 | 49 | 71 | 40.83% | 40.83% | 40.83% | 9.17 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 130 | 47 | 83 | 36.15% | 36.15% | 36.15% | 13.85 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 120 | 41 | 79 | 34.17% | 34.17% | 34.17% | 15.83 pp | -38 | 6 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 118 | 64 | 54 | 54.24% | 54.24% | 54.24% | 4.24 pp | 10 | 10 | 1.00 |
| BTC Market Hours | rf | RandomForest | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 10 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 10 | -0.80 |
| BTC Market Hours | transformer | Transformer | 118 | 49 | 69 | 41.53% | 41.53% | 41.53% | 8.47 pp | -20 | 10 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 118 | 47 | 71 | 39.83% | 39.83% | 39.83% | 10.17 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 118 | 46 | 72 | 38.98% | 38.98% | 38.98% | 11.02 pp | -26 | 10 | -2.60 |

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
| Consolidated Hourly | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 7 | 2 | 77.78% | 77.78% | 77.78% | 27.78 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
