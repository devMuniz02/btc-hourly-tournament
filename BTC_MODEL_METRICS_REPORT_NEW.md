# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T13:15:22.738738+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 194 | 134 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 230 | 170 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 302 | 158 | 144 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 302 | 158 | 144 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 134 | 69 | 65 | 51.49% | 51.49% | 51.49% | 1.49 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 158 | 83 | 75 | 52.53% | 52.53% | 52.53% | 2.53 pp | 8 | 13 | 0.62 |
| BTC Hourly | transformer | Transformer | 134 | 67 | 67 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 160 | 79 | 81 | 49.38% | 49.38% | 49.38% | 0.62 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 158 | 77 | 81 | 48.73% | 48.73% | 48.73% | 1.27 pp | -4 | 14 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 158 | 75 | 83 | 47.47% | 47.47% | 47.47% | 2.53 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| BTC Daily | nn | NN | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| BTC Market Hours | rf | RandomForest | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| BTC Market Hours | transformer | Transformer | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 158 | 70 | 88 | 44.30% | 44.30% | 44.30% | 5.70 pp | -18 | 13 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 14 | -2.00 |
| BTC Market Hours | lstm | LSTM | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |
| BTC Market Hours | xgb | XGBoost | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |
| BTC Hourly | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 6 | -2.33 |
| BTC Daily | transformer | Transformer | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Market Hours Daily | lstm | LSTM | 158 | 60 | 98 | 37.97% | 37.97% | 37.97% | 12.03 pp | -38 | 14 | -2.71 |
| BTC Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 170 | 64 | 106 | 37.65% | 37.65% | 37.65% | 12.35 pp | -42 | 8 | -5.25 |
| BTC Hourly | xgb | XGBoost | 134 | 50 | 84 | 37.31% | 37.31% | 37.31% | 12.69 pp | -34 | 6 | -5.67 |
| BTC Daily | lstm | LSTM | 160 | 60 | 100 | 37.50% | 37.50% | 37.50% | 12.50 pp | -40 | 7 | -5.71 |
| BTC Hourly | lstm | LSTM | 134 | 48 | 86 | 35.82% | 35.82% | 35.82% | 14.18 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 134 | 69 | 65 | 51.49% | 51.49% | 51.49% | 1.49 pp | 4 | 6 | 0.67 |
| BTC Hourly | transformer | Transformer | 134 | 67 | 67 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 6 | -2.33 |
| BTC Hourly | rf | RandomForest | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 134 | 50 | 84 | 37.31% | 37.31% | 37.31% | 12.69 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 134 | 48 | 86 | 35.82% | 35.82% | 35.82% | 14.18 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 160 | 79 | 81 | 49.38% | 49.38% | 49.38% | 0.62 pp | -2 | 7 | -0.29 |
| BTC Daily | nn | NN | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 7 | -1.14 |
| BTC Daily | transformer | Transformer | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 7 | -3.14 |
| BTC Daily | xgb | XGBoost | 170 | 64 | 106 | 37.65% | 37.65% | 37.65% | 12.35 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 160 | 60 | 100 | 37.50% | 37.50% | 37.50% | 12.50 pp | -40 | 7 | -5.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 158 | 83 | 75 | 52.53% | 52.53% | 52.53% | 2.53 pp | 8 | 13 | 0.62 |
| BTC Market Hours | rf | RandomForest | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| BTC Market Hours | transformer | Transformer | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 158 | 70 | 88 | 44.30% | 44.30% | 44.30% | 5.70 pp | -18 | 13 | -1.38 |
| BTC Market Hours | lstm | LSTM | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |
| BTC Market Hours | xgb | XGBoost | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 158 | 77 | 81 | 48.73% | 48.73% | 48.73% | 1.27 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 158 | 75 | 83 | 47.47% | 47.47% | 47.47% | 2.53 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 14 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 158 | 60 | 98 | 37.97% | 37.97% | 37.97% | 12.03 pp | -38 | 14 | -2.71 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
