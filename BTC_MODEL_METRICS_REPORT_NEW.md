# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T05:59:53.413183+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 190 | 130 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 225 | 165 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 296 | 153 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 296 | 153 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| BTC Market Hours | nn | NN | 153 | 80 | 73 | 52.29% | 52.29% | 52.29% | 2.29 pp | 7 | 12 | 0.58 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 13 | -0.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 155 | 75 | 80 | 48.39% | 48.39% | 48.39% | 1.61 pp | -5 | 7 | -0.71 |
| BTC Market Hours | transformer | Transformer | 153 | 70 | 83 | 45.75% | 45.75% | 45.75% | 4.25 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 13 | -1.15 |
| BTC Market Hours | rf | RandomForest | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 153 | 66 | 87 | 43.14% | 43.14% | 43.14% | 6.86 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 13 | -2.08 |
| BTC Market Hours | xgb | XGBoost | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 12 | -2.25 |
| Consolidated Hourly | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| BTC Hourly | nn | NN | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Daily | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 7 | -2.71 |
| BTC Daily | transformer | Transformer | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 7 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 153 | 56 | 97 | 36.60% | 36.60% | 36.60% | 13.40 pp | -41 | 13 | -3.15 |
| BTC Hourly | rf | RandomForest | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 7 | -3.86 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 130 | 49 | 81 | 37.69% | 37.69% | 37.69% | 12.31 pp | -32 | 6 | -5.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 165 | 58 | 107 | 35.15% | 35.15% | 35.15% | 14.85 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 130 | 46 | 84 | 35.38% | 35.38% | 35.38% | 14.62 pp | -38 | 6 | -6.33 |
| BTC Daily | lstm | LSTM | 155 | 55 | 100 | 35.48% | 35.48% | 35.48% | 14.52 pp | -45 | 7 | -6.43 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | nn | NN | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Hourly | rf | RandomForest | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 130 | 49 | 81 | 37.69% | 37.69% | 37.69% | 12.31 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 130 | 46 | 84 | 35.38% | 35.38% | 35.38% | 14.62 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 155 | 75 | 80 | 48.39% | 48.39% | 48.39% | 1.61 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 7 | -2.71 |
| BTC Daily | transformer | Transformer | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 7 | -3.00 |
| BTC Daily | rf | RandomForest | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 7 | -3.86 |
| BTC Daily | xgb | XGBoost | 165 | 58 | 107 | 35.15% | 35.15% | 35.15% | 14.85 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 155 | 55 | 100 | 35.48% | 35.48% | 35.48% | 14.52 pp | -45 | 7 | -6.43 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 153 | 80 | 73 | 52.29% | 52.29% | 52.29% | 2.29 pp | 7 | 12 | 0.58 |
| BTC Market Hours | transformer | Transformer | 153 | 70 | 83 | 45.75% | 45.75% | 45.75% | 4.25 pp | -13 | 12 | -1.08 |
| BTC Market Hours | rf | RandomForest | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 12 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 12 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 153 | 64 | 89 | 41.83% | 41.83% | 41.83% | 8.17 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 12 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 153 | 72 | 81 | 47.06% | 47.06% | 47.06% | 2.94 pp | -9 | 13 | -0.69 |
| BTC Market Hours Daily | nn | NN | 153 | 69 | 84 | 45.10% | 45.10% | 45.10% | 4.90 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 153 | 66 | 87 | 43.14% | 43.14% | 43.14% | 6.86 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 153 | 56 | 97 | 36.60% | 36.60% | 36.60% | 13.40 pp | -41 | 13 | -3.15 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
