# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T00:03:28.418786+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 169 | 109 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 205 | 145 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 262 | 133 | 129 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 262 | 133 | 129 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 109 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 110 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| BTC Market Hours | rf | RandomForest | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| BTC Daily | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 5 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |
| BTC Hourly | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 6 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 12 | -1.92 |
| BTC Daily | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 12 | -2.58 |
| BTC Market Hours | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 11 | -2.82 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 135 | 56 | 79 | 41.48% | 41.48% | 41.48% | 8.52 pp | -23 | 6 | -3.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| BTC Hourly | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 145 | 52 | 93 | 35.86% | 35.86% | 35.86% | 14.14 pp | -41 | 7 | -5.86 |
| BTC Hourly | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 5 | -6.60 |
| BTC Daily | lstm | LSTM | 135 | 47 | 88 | 34.81% | 34.81% | 34.81% | 15.19 pp | -41 | 6 | -6.83 |
| BTC Hourly | lstm | LSTM | 109 | 34 | 75 | 31.19% | 31.19% | 31.19% | 18.81 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Hourly | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 5 | -6.60 |
| BTC Hourly | lstm | LSTM | 109 | 34 | 75 | 31.19% | 31.19% | 31.19% | 18.81 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 135 | 65 | 70 | 48.15% | 48.15% | 48.15% | 1.85 pp | -5 | 6 | -0.83 |
| BTC Daily | nn | NN | 135 | 62 | 73 | 45.93% | 45.93% | 45.93% | 4.07 pp | -11 | 6 | -1.83 |
| BTC Daily | transformer | Transformer | 135 | 60 | 75 | 44.44% | 44.44% | 44.44% | 5.56 pp | -15 | 6 | -2.50 |
| BTC Daily | rf | RandomForest | 135 | 56 | 79 | 41.48% | 41.48% | 41.48% | 8.52 pp | -23 | 6 | -3.83 |
| BTC Daily | xgb | XGBoost | 145 | 52 | 93 | 35.86% | 35.86% | 35.86% | 14.14 pp | -41 | 7 | -5.86 |
| BTC Daily | lstm | LSTM | 135 | 47 | 88 | 34.81% | 34.81% | 34.81% | 15.19 pp | -41 | 6 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| BTC Market Hours | rf | RandomForest | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 11 | -1.00 |
| BTC Market Hours | transformer | Transformer | 133 | 58 | 75 | 43.61% | 43.61% | 43.61% | 6.39 pp | -17 | 11 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Market Hours | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 11 | -2.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | rf | RandomForest | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 133 | 61 | 72 | 45.86% | 45.86% | 45.86% | 4.14 pp | -11 | 12 | -0.92 |
| BTC Market Hours Daily | nn | NN | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | xgb | XGBoost | 133 | 55 | 78 | 41.35% | 41.35% | 41.35% | 8.65 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 133 | 51 | 82 | 38.35% | 38.35% | 38.35% | 11.65 pp | -31 | 12 | -2.58 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 10 | -1.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
