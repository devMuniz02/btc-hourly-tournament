# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T16:34:40.268436+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 148 | 88 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 184 | 124 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 15:00:00+00:00 | 220 | 112 | 108 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 15:00:00+00:00 | 220 | 112 | 108 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 112 | 60 | 52 | 53.57% | 53.57% | 53.57% | 3.57 pp | 8 | 9 | 0.89 |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| BTC Hourly | transformer | Transformer | 88 | 44 | 44 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 112 | 53 | 59 | 47.32% | 47.32% | 47.32% | 2.68 pp | -6 | 9 | -0.67 |
| BTC Market Hours | rf | RandomForest | 112 | 53 | 59 | 47.32% | 47.32% | 47.32% | 2.68 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | transformer | Transformer | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 10 | -0.80 |
| BTC Hourly | nn | NN | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 4 | -1.00 |
| BTC Daily | nn | NN | 114 | 54 | 60 | 47.37% | 47.37% | 47.37% | 2.63 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 112 | 49 | 63 | 43.75% | 43.75% | 43.75% | 6.25 pp | -14 | 10 | -1.40 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 88 | 41 | 47 | 46.59% | 46.59% | 46.59% | 3.41 pp | -6 | 4 | -1.50 |
| BTC Market Hours | transformer | Transformer | 112 | 48 | 64 | 42.86% | 42.86% | 42.86% | 7.14 pp | -16 | 9 | -1.78 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 112 | 44 | 68 | 39.29% | 39.29% | 39.29% | 10.71 pp | -24 | 10 | -2.40 |
| BTC Market Hours | xgb | XGBoost | 112 | 45 | 67 | 40.18% | 40.18% | 40.18% | 9.82 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 112 | 44 | 68 | 39.29% | 39.29% | 39.29% | 10.71 pp | -24 | 9 | -2.67 |
| BTC Market Hours Daily | lstm | LSTM | 112 | 42 | 70 | 37.50% | 37.50% | 37.50% | 12.50 pp | -28 | 10 | -2.80 |
| BTC Daily | rf | RandomForest | 114 | 46 | 68 | 40.35% | 40.35% | 40.35% | 9.65 pp | -22 | 6 | -3.67 |
| BTC Hourly | rf | RandomForest | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 4 | -4.50 |
| BTC Daily | xgb | XGBoost | 124 | 45 | 79 | 36.29% | 36.29% | 36.29% | 13.71 pp | -34 | 7 | -4.86 |
| BTC Daily | lstm | LSTM | 114 | 40 | 74 | 35.09% | 35.09% | 35.09% | 14.91 pp | -34 | 6 | -5.67 |
| BTC Hourly | xgb | XGBoost | 88 | 29 | 59 | 32.95% | 32.95% | 32.95% | 17.05 pp | -30 | 4 | -7.50 |
| BTC Hourly | lstm | LSTM | 88 | 28 | 60 | 31.82% | 31.82% | 31.82% | 18.18 pp | -32 | 4 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 88 | 44 | 44 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | nn | NN | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 88 | 41 | 47 | 46.59% | 46.59% | 46.59% | 3.41 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 4 | -4.50 |
| BTC Hourly | xgb | XGBoost | 88 | 29 | 59 | 32.95% | 32.95% | 32.95% | 17.05 pp | -30 | 4 | -7.50 |
| BTC Hourly | lstm | LSTM | 88 | 28 | 60 | 31.82% | 31.82% | 31.82% | 18.18 pp | -32 | 4 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 6 | -0.33 |
| BTC Daily | nn | NN | 114 | 54 | 60 | 47.37% | 47.37% | 47.37% | 2.63 pp | -6 | 6 | -1.00 |
| BTC Daily | transformer | Transformer | 114 | 50 | 64 | 43.86% | 43.86% | 43.86% | 6.14 pp | -14 | 6 | -2.33 |
| BTC Daily | rf | RandomForest | 114 | 46 | 68 | 40.35% | 40.35% | 40.35% | 9.65 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 124 | 45 | 79 | 36.29% | 36.29% | 36.29% | 13.71 pp | -34 | 7 | -4.86 |
| BTC Daily | lstm | LSTM | 114 | 40 | 74 | 35.09% | 35.09% | 35.09% | 14.91 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 112 | 60 | 52 | 53.57% | 53.57% | 53.57% | 3.57 pp | 8 | 9 | 0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 112 | 53 | 59 | 47.32% | 47.32% | 47.32% | 2.68 pp | -6 | 9 | -0.67 |
| BTC Market Hours | rf | RandomForest | 112 | 53 | 59 | 47.32% | 47.32% | 47.32% | 2.68 pp | -6 | 9 | -0.67 |
| BTC Market Hours | transformer | Transformer | 112 | 48 | 64 | 42.86% | 42.86% | 42.86% | 7.14 pp | -16 | 9 | -1.78 |
| BTC Market Hours | xgb | XGBoost | 112 | 45 | 67 | 40.18% | 40.18% | 40.18% | 9.82 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 112 | 44 | 68 | 39.29% | 39.29% | 39.29% | 10.71 pp | -24 | 9 | -2.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 112 | 57 | 55 | 50.89% | 50.89% | 50.89% | 0.89 pp | 2 | 10 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 112 | 52 | 60 | 46.43% | 46.43% | 46.43% | 3.57 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 112 | 50 | 62 | 44.64% | 44.64% | 44.64% | 5.36 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 112 | 49 | 63 | 43.75% | 43.75% | 43.75% | 6.25 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 112 | 44 | 68 | 39.29% | 39.29% | 39.29% | 10.71 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 112 | 42 | 70 | 37.50% | 37.50% | 37.50% | 12.50 pp | -28 | 10 | -2.80 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
