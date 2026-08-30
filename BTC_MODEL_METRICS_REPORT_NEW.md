# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T15:19:54.638142+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 130 | 70 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 166 | 106 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 188 | 94 | 94 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 188 | 94 | 94 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 14:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 14:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 14:00:00+00:00 | 75 | 1 | 74 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 14:00:00+00:00 | 75 | 1 | 74 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 94 | 51 | 43 | 54.26% | 54.26% | 54.26% | 4.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 8 | 0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 48 | 46 | 51.06% | 51.06% | 51.06% | 1.06 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| BTC Market Hours | rf | RandomForest | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 8 | -0.50 |
| BTC Hourly | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | nn | NN | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 9 | -1.56 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 96 | 43 | 53 | 44.79% | 44.79% | 44.79% | 5.21 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 9 | -2.22 |
| Consolidated Hourly | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 8 | -2.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 8 | -2.38 |
| BTC Market Hours | transformer | Transformer | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 8 | -2.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 3 | -2.67 |
| BTC Market Hours Daily | xgb | XGBoost | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 9 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 5 | -4.00 |
| BTC Hourly | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 96 | 34 | 62 | 35.42% | 35.42% | 35.42% | 14.58 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 70 | 22 | 48 | 31.43% | 31.43% | 31.43% | 18.57 pp | -26 | 3 | -8.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 70 | 37 | 33 | 52.86% | 52.86% | 52.86% | 2.86 pp | 4 | 3 | 1.33 |
| BTC Hourly | nn | NN | 70 | 34 | 36 | 48.57% | 48.57% | 48.57% | 1.43 pp | -2 | 3 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 3 | -2.67 |
| BTC Hourly | rf | RandomForest | 70 | 28 | 42 | 40.00% | 40.00% | 40.00% | 10.00 pp | -14 | 3 | -4.67 |
| BTC Hourly | lstm | LSTM | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 70 | 22 | 48 | 31.43% | 31.43% | 31.43% | 18.57 pp | -26 | 3 | -8.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 96 | 43 | 53 | 44.79% | 44.79% | 44.79% | 5.21 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 96 | 34 | 62 | 35.42% | 35.42% | 35.42% | 14.58 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 94 | 51 | 43 | 54.26% | 54.26% | 54.26% | 4.26 pp | 8 | 8 | 1.00 |
| BTC Market Hours | rf | RandomForest | 94 | 46 | 48 | 48.94% | 48.94% | 48.94% | 1.06 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 94 | 45 | 49 | 47.87% | 47.87% | 47.87% | 2.13 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 94 | 39 | 55 | 41.49% | 41.49% | 41.49% | 8.51 pp | -16 | 8 | -2.00 |
| BTC Market Hours | transformer | Transformer | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 48 | 46 | 51.06% | 51.06% | 51.06% | 1.06 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 94 | 43 | 51 | 45.74% | 45.74% | 45.74% | 4.26 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 94 | 37 | 57 | 39.36% | 39.36% | 39.36% | 10.64 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 9 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 8 | -2.38 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 8 | 0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 8 | -2.38 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
