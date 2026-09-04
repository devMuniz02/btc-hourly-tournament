# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T05:55:05.158447+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 205 | 145 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 241 | 181 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 325 | 169 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 325 | 169 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 143 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 17:00:00+00:00 | 143 | 33 | 110 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 169 | 88 | 81 | 52.07% | 52.07% | 52.07% | 2.07 pp | 7 | 13 | 0.54 |
| BTC Hourly | transformer | Transformer | 145 | 72 | 73 | 49.66% | 49.66% | 49.66% | 0.34 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 14 | -0.21 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 169 | 81 | 88 | 47.93% | 47.93% | 47.93% | 2.07 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 169 | 79 | 90 | 46.75% | 46.75% | 46.75% | 3.25 pp | -11 | 13 | -0.85 |
| BTC Market Hours Daily | nn | NN | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 14 | -0.93 |
| BTC Market Hours | transformer | Transformer | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 14 | -1.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 8 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 169 | 71 | 98 | 42.01% | 42.01% | 42.01% | 7.99 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 171 | 77 | 94 | 45.03% | 45.03% | 45.03% | 4.97 pp | -17 | 8 | -2.12 |
| BTC Market Hours | lstm | LSTM | 169 | 70 | 99 | 41.42% | 41.42% | 41.42% | 8.58 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | transformer | Transformer | 171 | 75 | 96 | 43.86% | 43.86% | 43.86% | 6.14 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | lstm | LSTM | 169 | 66 | 103 | 39.05% | 39.05% | 39.05% | 10.95 pp | -37 | 14 | -2.64 |
| BTC Hourly | nn | NN | 145 | 63 | 82 | 43.45% | 43.45% | 43.45% | 6.55 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| BTC Hourly | rf | RandomForest | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 8 | -3.62 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 181 | 68 | 113 | 37.57% | 37.57% | 37.57% | 12.43 pp | -45 | 9 | -5.00 |
| BTC Hourly | lstm | LSTM | 145 | 53 | 92 | 36.55% | 36.55% | 36.55% | 13.45 pp | -39 | 7 | -5.57 |
| BTC Hourly | xgb | XGBoost | 145 | 53 | 92 | 36.55% | 36.55% | 36.55% | 13.45 pp | -39 | 7 | -5.57 |
| BTC Daily | lstm | LSTM | 171 | 60 | 111 | 35.09% | 35.09% | 35.09% | 14.91 pp | -51 | 8 | -6.38 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 145 | 72 | 73 | 49.66% | 49.66% | 49.66% | 0.34 pp | -1 | 7 | -0.14 |
| BTC Hourly | nn | NN | 145 | 63 | 82 | 43.45% | 43.45% | 43.45% | 6.55 pp | -19 | 7 | -2.71 |
| BTC Hourly | rf | RandomForest | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 145 | 53 | 92 | 36.55% | 36.55% | 36.55% | 13.45 pp | -39 | 7 | -5.57 |
| BTC Hourly | xgb | XGBoost | 145 | 53 | 92 | 36.55% | 36.55% | 36.55% | 13.45 pp | -39 | 7 | -5.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 8 | -1.62 |
| BTC Daily | nn | NN | 171 | 77 | 94 | 45.03% | 45.03% | 45.03% | 4.97 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 171 | 75 | 96 | 43.86% | 43.86% | 43.86% | 6.14 pp | -21 | 8 | -2.62 |
| BTC Daily | rf | RandomForest | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 8 | -3.62 |
| BTC Daily | xgb | XGBoost | 181 | 68 | 113 | 37.57% | 37.57% | 37.57% | 12.43 pp | -45 | 9 | -5.00 |
| BTC Daily | lstm | LSTM | 171 | 60 | 111 | 35.09% | 35.09% | 35.09% | 14.91 pp | -51 | 8 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 169 | 88 | 81 | 52.07% | 52.07% | 52.07% | 2.07 pp | 7 | 13 | 0.54 |
| BTC Market Hours | rf | RandomForest | 169 | 79 | 90 | 46.75% | 46.75% | 46.75% | 3.25 pp | -11 | 13 | -0.85 |
| BTC Market Hours | transformer | Transformer | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 13 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 13 | -1.15 |
| BTC Market Hours | xgb | XGBoost | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 13 | -1.92 |
| BTC Market Hours | lstm | LSTM | 169 | 70 | 99 | 41.42% | 41.42% | 41.42% | 8.58 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 169 | 81 | 88 | 47.93% | 47.93% | 47.93% | 2.07 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 14 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 169 | 71 | 98 | 42.01% | 42.01% | 42.01% | 7.99 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 169 | 66 | 103 | 39.05% | 39.05% | 39.05% | 10.95 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 143 | 67 | 76 | 46.85% | 46.85% | 46.85% | 3.15 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 143 | 66 | 77 | 46.15% | 46.15% | 46.15% | 3.85 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 143 | 61 | 82 | 42.66% | 42.66% | 42.66% | 7.34 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 12 | 21 | 36.36% | 36.36% | 36.36% | 13.64 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 10 | 23 | 30.30% | 30.30% | 30.30% | 19.70 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
