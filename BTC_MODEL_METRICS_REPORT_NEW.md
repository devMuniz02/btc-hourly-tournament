# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T08:53:47.450814+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 192 | 132 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 227 | 167 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 298 | 155 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 298 | 155 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 78 | 79 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 13 | -0.69 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | nn | NN | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Daily | nn | NN | 157 | 74 | 83 | 47.13% | 47.13% | 47.13% | 2.87 pp | -9 | 7 | -1.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Hourly | nn | NN | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 157 | 70 | 87 | 44.59% | 44.59% | 44.59% | 5.41 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | lstm | LSTM | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 13 | -3.15 |
| BTC Daily | rf | RandomForest | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 132 | 56 | 76 | 42.42% | 42.42% | 42.42% | 7.58 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| BTC Daily | lstm | LSTM | 157 | 60 | 97 | 38.22% | 38.22% | 38.22% | 11.78 pp | -37 | 7 | -5.29 |
| BTC Daily | xgb | XGBoost | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 8 | -5.62 |
| BTC Hourly | xgb | XGBoost | 132 | 49 | 83 | 37.12% | 37.12% | 37.12% | 12.88 pp | -34 | 6 | -5.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| BTC Hourly | lstm | LSTM | 132 | 47 | 85 | 35.61% | 35.61% | 35.61% | 14.39 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 132 | 67 | 65 | 50.76% | 50.76% | 50.76% | 0.76 pp | 2 | 6 | 0.33 |
| BTC Hourly | nn | NN | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 6 | -2.33 |
| BTC Hourly | rf | RandomForest | 132 | 56 | 76 | 42.42% | 42.42% | 42.42% | 7.58 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 132 | 49 | 83 | 37.12% | 37.12% | 37.12% | 12.88 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 132 | 47 | 85 | 35.61% | 35.61% | 35.61% | 14.39 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 78 | 79 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 7 | -0.14 |
| BTC Daily | nn | NN | 157 | 74 | 83 | 47.13% | 47.13% | 47.13% | 2.87 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 157 | 70 | 87 | 44.59% | 44.59% | 44.59% | 5.41 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 7 | -3.29 |
| BTC Daily | lstm | LSTM | 157 | 60 | 97 | 38.22% | 38.22% | 38.22% | 11.78 pp | -37 | 7 | -5.29 |
| BTC Daily | xgb | XGBoost | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 8 | -5.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 13 | -0.69 |
| BTC Market Hours Daily | nn | NN | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 155 | 57 | 98 | 36.77% | 36.77% | 36.77% | 13.23 pp | -41 | 13 | -3.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
