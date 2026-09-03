# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T12:10:03.828747+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 301 | 158 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 300 | 157 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 131 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 131 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 27 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 00:00:00+00:00 | 131 | 27 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 158 | 83 | 75 | 52.53% | 52.53% | 52.53% | 2.53 pp | 8 | 13 | 0.62 |
| Consolidated Market Hours | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 134 | 67 | 67 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 13 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 160 | 79 | 81 | 49.38% | 49.38% | 49.38% | 0.62 pp | -2 | 7 | -0.29 |
| Consolidated Market Hours | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 157 | 75 | 82 | 47.77% | 47.77% | 47.77% | 2.23 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| BTC Market Hours | rf | RandomForest | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| BTC Market Hours | transformer | Transformer | 158 | 71 | 87 | 44.94% | 44.94% | 44.94% | 5.06 pp | -16 | 13 | -1.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 158 | 70 | 88 | 44.30% | 44.30% | 44.30% | 5.70 pp | -18 | 13 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 157 | 68 | 89 | 43.31% | 43.31% | 43.31% | 6.69 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 13 | -2.08 |
| Consolidated Hourly | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |
| BTC Market Hours | xgb | XGBoost | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 13 | -2.15 |
| BTC Hourly | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 6 | -2.33 |
| BTC Daily | transformer | Transformer | 160 | 71 | 89 | 44.38% | 44.38% | 44.38% | 5.63 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 157 | 59 | 98 | 37.58% | 37.58% | 37.58% | 12.42 pp | -39 | 13 | -3.00 |
| Consolidated Market Hours | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| BTC Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 134 | 57 | 77 | 42.54% | 42.54% | 42.54% | 7.46 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 170 | 64 | 106 | 37.65% | 37.65% | 37.65% | 12.35 pp | -42 | 8 | -5.25 |
| BTC Hourly | xgb | XGBoost | 134 | 50 | 84 | 37.31% | 37.31% | 37.31% | 12.69 pp | -34 | 6 | -5.67 |
| BTC Daily | lstm | LSTM | 160 | 60 | 100 | 37.50% | 37.50% | 37.50% | 12.50 pp | -40 | 7 | -5.71 |
| BTC Hourly | lstm | LSTM | 134 | 48 | 86 | 35.82% | 35.82% | 35.82% | 14.18 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 6 | 0.33 |
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
| BTC Market Hours Daily | transformer | Transformer | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 157 | 75 | 82 | 47.77% | 47.77% | 47.77% | 2.23 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 157 | 68 | 89 | 43.31% | 43.31% | 43.31% | 6.69 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 157 | 65 | 92 | 41.40% | 41.40% | 41.40% | 8.60 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 157 | 59 | 98 | 37.58% | 37.58% | 37.58% | 12.42 pp | -39 | 13 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 131 | 69 | 62 | 52.67% | 52.67% | 52.67% | 2.67 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 131 | 63 | 68 | 48.09% | 48.09% | 48.09% | 1.91 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 131 | 60 | 71 | 45.80% | 45.80% | 45.80% | 4.20 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 131 | 54 | 77 | 41.22% | 41.22% | 41.22% | 8.78 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 27 | 9 | 18 | 33.33% | 33.33% | 33.33% | 16.67 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
