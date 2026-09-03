# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T04:22:42.299218+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 188 | 128 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 224 | 164 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 295 | 152 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 295 | 152 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| BTC Market Hours | nn | NN | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 12 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| BTC Market Hours | rf | RandomForest | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | nn | NN | 152 | 68 | 84 | 44.74% | 44.74% | 44.74% | 5.26 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 152 | 68 | 84 | 44.74% | 44.74% | 44.74% | 5.26 pp | -16 | 12 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 13 | -1.54 |
| BTC Hourly | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 12 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 12 | -2.17 |
| BTC Daily | nn | NN | 154 | 69 | 85 | 44.81% | 44.81% | 44.81% | 5.19 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| BTC Daily | transformer | Transformer | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 152 | 56 | 96 | 36.84% | 36.84% | 36.84% | 13.16 pp | -40 | 13 | -3.08 |
| BTC Hourly | rf | RandomForest | 128 | 54 | 74 | 42.19% | 42.19% | 42.19% | 7.81 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 6 | -5.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 164 | 58 | 106 | 35.37% | 35.37% | 35.37% | 14.63 pp | -48 | 8 | -6.00 |
| BTC Daily | lstm | LSTM | 154 | 55 | 99 | 35.71% | 35.71% | 35.71% | 14.29 pp | -44 | 7 | -6.29 |
| BTC Hourly | lstm | LSTM | 128 | 45 | 83 | 35.16% | 35.16% | 35.16% | 14.84 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 128 | 64 | 64 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | nn | NN | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 6 | -2.00 |
| BTC Hourly | rf | RandomForest | 128 | 54 | 74 | 42.19% | 42.19% | 42.19% | 7.81 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 128 | 48 | 80 | 37.50% | 37.50% | 37.50% | 12.50 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 128 | 45 | 83 | 35.16% | 35.16% | 35.16% | 14.84 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 154 | 75 | 79 | 48.70% | 48.70% | 48.70% | 1.30 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 154 | 69 | 85 | 44.81% | 44.81% | 44.81% | 5.19 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 154 | 68 | 86 | 44.16% | 44.16% | 44.16% | 5.84 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 154 | 65 | 89 | 42.21% | 42.21% | 42.21% | 7.79 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 164 | 58 | 106 | 35.37% | 35.37% | 35.37% | 14.63 pp | -48 | 8 | -6.00 |
| BTC Daily | lstm | LSTM | 154 | 55 | 99 | 35.71% | 35.71% | 35.71% | 14.29 pp | -44 | 7 | -6.29 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 12 | -1.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 152 | 68 | 84 | 44.74% | 44.74% | 44.74% | 5.26 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 12 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 12 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 13 | -0.62 |
| BTC Market Hours Daily | nn | NN | 152 | 68 | 84 | 44.74% | 44.74% | 44.74% | 5.26 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 13 | -1.54 |
| BTC Market Hours Daily | xgb | XGBoost | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 152 | 56 | 96 | 36.84% | 36.84% | 36.84% | 13.16 pp | -40 | 13 | -3.08 |

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
