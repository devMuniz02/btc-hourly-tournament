# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T06:09:41.315750+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 158 | 98 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 194 | 134 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 239 | 122 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 238 | 121 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| BTC Market Hours | nn | NN | 122 | 64 | 58 | 52.46% | 52.46% | 52.46% | 2.46 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 124 | 62 | 62 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 121 | 60 | 61 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 11 | -0.09 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| BTC Market Hours | rf | RandomForest | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| BTC Hourly | nn | NN | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Hourly | transformer | Transformer | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 11 | -1.18 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 98 | 46 | 52 | 46.94% | 46.94% | 46.94% | 3.06 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 121 | 48 | 73 | 39.67% | 39.67% | 39.67% | 10.33 pp | -25 | 11 | -2.27 |
| BTC Market Hours | xgb | XGBoost | 122 | 49 | 73 | 40.16% | 40.16% | 40.16% | 9.84 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 121 | 46 | 75 | 38.02% | 38.02% | 38.02% | 11.98 pp | -29 | 11 | -2.64 |
| BTC Market Hours | lstm | LSTM | 122 | 47 | 75 | 38.52% | 38.52% | 38.52% | 11.48 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 124 | 52 | 72 | 41.94% | 41.94% | 41.94% | 8.06 pp | -20 | 6 | -3.33 |
| BTC Hourly | rf | RandomForest | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 5 | -4.00 |
| BTC Daily | xgb | XGBoost | 134 | 49 | 85 | 36.57% | 36.57% | 36.57% | 13.43 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 124 | 43 | 81 | 34.68% | 34.68% | 34.68% | 15.32 pp | -38 | 6 | -6.33 |
| BTC Hourly | xgb | XGBoost | 98 | 32 | 66 | 32.65% | 32.65% | 32.65% | 17.35 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 98 | 31 | 67 | 31.63% | 31.63% | 31.63% | 18.37 pp | -36 | 5 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Hourly | transformer | Transformer | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 98 | 46 | 52 | 46.94% | 46.94% | 46.94% | 3.06 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 98 | 32 | 66 | 32.65% | 32.65% | 32.65% | 17.35 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 98 | 31 | 67 | 31.63% | 31.63% | 31.63% | 18.37 pp | -36 | 5 | -7.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 124 | 62 | 62 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Daily | rf | RandomForest | 124 | 52 | 72 | 41.94% | 41.94% | 41.94% | 8.06 pp | -20 | 6 | -3.33 |
| BTC Daily | xgb | XGBoost | 134 | 49 | 85 | 36.57% | 36.57% | 36.57% | 13.43 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 124 | 43 | 81 | 34.68% | 34.68% | 34.68% | 15.32 pp | -38 | 6 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 122 | 64 | 58 | 52.46% | 52.46% | 52.46% | 2.46 pp | 6 | 10 | 0.60 |
| BTC Market Hours | rf | RandomForest | 122 | 59 | 63 | 48.36% | 48.36% | 48.36% | 1.64 pp | -4 | 10 | -0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 122 | 56 | 66 | 45.90% | 45.90% | 45.90% | 4.10 pp | -10 | 10 | -1.00 |
| BTC Market Hours | transformer | Transformer | 122 | 52 | 70 | 42.62% | 42.62% | 42.62% | 7.38 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 122 | 49 | 73 | 40.16% | 40.16% | 40.16% | 9.84 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 122 | 47 | 75 | 38.52% | 38.52% | 38.52% | 11.48 pp | -28 | 10 | -2.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 121 | 60 | 61 | 49.59% | 49.59% | 49.59% | 0.41 pp | -1 | 11 | -0.09 |
| BTC Market Hours Daily | rf | RandomForest | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 121 | 55 | 66 | 45.45% | 45.45% | 45.45% | 4.55 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 121 | 54 | 67 | 44.63% | 44.63% | 44.63% | 5.37 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | xgb | XGBoost | 121 | 48 | 73 | 39.67% | 39.67% | 39.67% | 10.33 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 121 | 46 | 75 | 38.02% | 38.02% | 38.02% | 11.98 pp | -29 | 11 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
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
