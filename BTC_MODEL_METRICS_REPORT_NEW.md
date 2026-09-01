# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T16:25:52.371572+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 164 | 104 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 200 | 140 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 15:00:00+00:00 | 249 | 128 | 121 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 15:00:00+00:00 | 249 | 128 | 121 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| BTC Market Hours | nn | NN | 128 | 68 | 60 | 53.12% | 53.12% | 53.12% | 3.12 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 104 | 51 | 53 | 49.04% | 49.04% | 49.04% | 0.96 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 11 | -0.55 |
| BTC Market Hours | rf | RandomForest | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 128 | 59 | 69 | 46.09% | 46.09% | 46.09% | 3.91 pp | -10 | 11 | -0.91 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 11 | -1.09 |
| BTC Hourly | nn | NN | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 10 | -1.20 |
| BTC Daily | nn | NN | 130 | 61 | 69 | 46.92% | 46.92% | 46.92% | 3.08 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| BTC Market Hours | transformer | Transformer | 128 | 55 | 73 | 42.97% | 42.97% | 42.97% | 7.03 pp | -18 | 10 | -1.80 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| BTC Daily | transformer | Transformer | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 128 | 52 | 76 | 40.62% | 40.62% | 40.62% | 9.38 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 128 | 50 | 78 | 39.06% | 39.06% | 39.06% | 10.94 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 11 | -2.91 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Market Hours | lstm | LSTM | 128 | 49 | 79 | 38.28% | 38.28% | 38.28% | 11.72 pp | -30 | 10 | -3.00 |
| BTC Hourly | rf | RandomForest | 104 | 43 | 61 | 41.35% | 41.35% | 41.35% | 8.65 pp | -18 | 5 | -3.60 |
| BTC Daily | rf | RandomForest | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Daily | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 7 | -5.43 |
| BTC Hourly | xgb | XGBoost | 104 | 37 | 67 | 35.58% | 35.58% | 35.58% | 14.42 pp | -30 | 5 | -6.00 |
| BTC Daily | lstm | LSTM | 130 | 45 | 85 | 34.62% | 34.62% | 34.62% | 15.38 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 104 | 32 | 72 | 30.77% | 30.77% | 30.77% | 19.23 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 104 | 51 | 53 | 49.04% | 49.04% | 49.04% | 0.96 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 104 | 43 | 61 | 41.35% | 41.35% | 41.35% | 8.65 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 104 | 37 | 67 | 35.58% | 35.58% | 35.58% | 14.42 pp | -30 | 5 | -6.00 |
| BTC Hourly | lstm | LSTM | 104 | 32 | 72 | 30.77% | 30.77% | 30.77% | 19.23 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | nn | NN | 130 | 61 | 69 | 46.92% | 46.92% | 46.92% | 3.08 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Daily | rf | RandomForest | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 130 | 45 | 85 | 34.62% | 34.62% | 34.62% | 15.38 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 128 | 68 | 60 | 53.12% | 53.12% | 53.12% | 3.12 pp | 8 | 10 | 0.80 |
| BTC Market Hours | rf | RandomForest | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 10 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 10 | -1.20 |
| BTC Market Hours | transformer | Transformer | 128 | 55 | 73 | 42.97% | 42.97% | 42.97% | 7.03 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 128 | 52 | 76 | 40.62% | 40.62% | 40.62% | 9.38 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 128 | 49 | 79 | 38.28% | 38.28% | 38.28% | 11.72 pp | -30 | 10 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 11 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 128 | 59 | 69 | 46.09% | 46.09% | 46.09% | 3.91 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | xgb | XGBoost | 128 | 50 | 78 | 39.06% | 39.06% | 39.06% | 10.94 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 11 | -2.91 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
