# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T03:53:24.607031+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 172 | 112 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 208 | 148 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 266 | 136 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 266 | 136 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 112 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T13:00:00+00:00 | 113 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 136 | 71 | 65 | 52.21% | 52.21% | 52.21% | 2.21 pp | 6 | 11 | 0.55 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 136 | 67 | 69 | 49.26% | 49.26% | 49.26% | 0.74 pp | -2 | 12 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 7 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| BTC Market Hours | rf | RandomForest | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 136 | 64 | 72 | 47.06% | 47.06% | 47.06% | 2.94 pp | -8 | 12 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 136 | 63 | 73 | 46.32% | 46.32% | 46.32% | 3.68 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | rf | RandomForest | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| BTC Market Hours | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 5 | -1.60 |
| BTC Daily | nn | NN | 138 | 63 | 75 | 45.65% | 45.65% | 45.65% | 4.35 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |
| BTC Market Hours Daily | xgb | XGBoost | 136 | 56 | 80 | 41.18% | 41.18% | 41.18% | 8.82 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Daily | transformer | Transformer | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 7 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 136 | 52 | 84 | 38.24% | 38.24% | 38.24% | 11.76 pp | -32 | 12 | -2.67 |
| BTC Market Hours | lstm | LSTM | 136 | 52 | 84 | 38.24% | 38.24% | 38.24% | 11.76 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| BTC Hourly | rf | RandomForest | 112 | 46 | 66 | 41.07% | 41.07% | 41.07% | 8.93 pp | -20 | 5 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 8 | -5.00 |
| BTC Daily | lstm | LSTM | 138 | 49 | 89 | 35.51% | 35.51% | 35.51% | 14.49 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 112 | 39 | 73 | 34.82% | 34.82% | 34.82% | 15.18 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 112 | 36 | 76 | 32.14% | 32.14% | 32.14% | 17.86 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 5 | -0.80 |
| BTC Hourly | nn | NN | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 112 | 46 | 66 | 41.07% | 41.07% | 41.07% | 8.93 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 112 | 39 | 73 | 34.82% | 34.82% | 34.82% | 15.18 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 112 | 36 | 76 | 32.14% | 32.14% | 32.14% | 17.86 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 138 | 68 | 70 | 49.28% | 49.28% | 49.28% | 0.72 pp | -2 | 7 | -0.29 |
| BTC Daily | nn | NN | 138 | 63 | 75 | 45.65% | 45.65% | 45.65% | 4.35 pp | -12 | 7 | -1.71 |
| BTC Daily | transformer | Transformer | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 7 | -3.14 |
| BTC Daily | xgb | XGBoost | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 8 | -5.00 |
| BTC Daily | lstm | LSTM | 138 | 49 | 89 | 35.51% | 35.51% | 35.51% | 14.49 pp | -40 | 7 | -5.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 136 | 71 | 65 | 52.21% | 52.21% | 52.21% | 2.21 pp | 6 | 11 | 0.55 |
| BTC Market Hours | rf | RandomForest | 136 | 65 | 71 | 47.79% | 47.79% | 47.79% | 2.21 pp | -6 | 11 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 136 | 63 | 73 | 46.32% | 46.32% | 46.32% | 3.68 pp | -10 | 11 | -0.91 |
| BTC Market Hours | transformer | Transformer | 136 | 60 | 76 | 44.12% | 44.12% | 44.12% | 5.88 pp | -16 | 11 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 11 | -1.82 |
| BTC Market Hours | lstm | LSTM | 136 | 52 | 84 | 38.24% | 38.24% | 38.24% | 11.76 pp | -32 | 11 | -2.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 136 | 67 | 69 | 49.26% | 49.26% | 49.26% | 0.74 pp | -2 | 12 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 136 | 64 | 72 | 47.06% | 47.06% | 47.06% | 2.94 pp | -8 | 12 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 136 | 62 | 74 | 45.59% | 45.59% | 45.59% | 4.41 pp | -12 | 12 | -1.00 |
| BTC Market Hours Daily | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 136 | 56 | 80 | 41.18% | 41.18% | 41.18% | 8.82 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 136 | 52 | 84 | 38.24% | 38.24% | 38.24% | 11.76 pp | -32 | 12 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 112 | 51 | 61 | 45.54% | 45.54% | 45.54% | 4.46 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
