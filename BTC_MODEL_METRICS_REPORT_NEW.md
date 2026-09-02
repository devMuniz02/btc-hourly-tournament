# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T22:28:00.430844+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 184 | 124 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 220 | 160 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 21:00:00+00:00 | 288 | 148 | 140 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 21:00:00+00:00 | 288 | 148 | 140 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| BTC Market Hours | nn | NN | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 12 | 0.50 |
| BTC Hourly | transformer | Transformer | 124 | 62 | 62 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 13 | -0.46 |
| Consolidated Hourly | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 148 | 70 | 78 | 47.30% | 47.30% | 47.30% | 2.70 pp | -8 | 13 | -0.62 |
| BTC Market Hours | rf | RandomForest | 148 | 69 | 79 | 46.62% | 46.62% | 46.62% | 3.38 pp | -10 | 12 | -0.83 |
| BTC Market Hours Daily | nn | NN | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 148 | 66 | 82 | 44.59% | 44.59% | 44.59% | 5.41 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| BTC Market Hours | transformer | Transformer | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 148 | 62 | 86 | 41.89% | 41.89% | 41.89% | 8.11 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 13 | -2.00 |
| BTC Daily | nn | NN | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 148 | 59 | 89 | 39.86% | 39.86% | 39.86% | 10.14 pp | -30 | 12 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 148 | 55 | 93 | 37.16% | 37.16% | 37.16% | 12.84 pp | -38 | 13 | -2.92 |
| BTC Daily | rf | RandomForest | 150 | 63 | 87 | 42.00% | 42.00% | 42.00% | 8.00 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 124 | 51 | 73 | 41.13% | 41.13% | 41.13% | 8.87 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 124 | 46 | 78 | 37.10% | 37.10% | 37.10% | 12.90 pp | -32 | 6 | -5.33 |
| BTC Daily | xgb | XGBoost | 160 | 58 | 102 | 36.25% | 36.25% | 36.25% | 13.75 pp | -44 | 8 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| BTC Daily | lstm | LSTM | 150 | 53 | 97 | 35.33% | 35.33% | 35.33% | 14.67 pp | -44 | 7 | -6.29 |
| BTC Hourly | lstm | LSTM | 124 | 42 | 82 | 33.87% | 33.87% | 33.87% | 16.13 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 124 | 62 | 62 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Hourly | rf | RandomForest | 124 | 51 | 73 | 41.13% | 41.13% | 41.13% | 8.87 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 124 | 46 | 78 | 37.10% | 37.10% | 37.10% | 12.90 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 124 | 42 | 82 | 33.87% | 33.87% | 33.87% | 16.13 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 150 | 63 | 87 | 42.00% | 42.00% | 42.00% | 8.00 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 160 | 58 | 102 | 36.25% | 36.25% | 36.25% | 13.75 pp | -44 | 8 | -5.50 |
| BTC Daily | lstm | LSTM | 150 | 53 | 97 | 35.33% | 35.33% | 35.33% | 14.67 pp | -44 | 7 | -6.29 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 148 | 69 | 79 | 46.62% | 46.62% | 46.62% | 3.38 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 12 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 148 | 62 | 86 | 41.89% | 41.89% | 41.89% | 8.11 pp | -24 | 12 | -2.00 |
| BTC Market Hours | lstm | LSTM | 148 | 59 | 89 | 39.86% | 39.86% | 39.86% | 10.14 pp | -30 | 12 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 148 | 70 | 78 | 47.30% | 47.30% | 47.30% | 2.70 pp | -8 | 13 | -0.62 |
| BTC Market Hours Daily | nn | NN | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 148 | 66 | 82 | 44.59% | 44.59% | 44.59% | 5.41 pp | -16 | 13 | -1.23 |
| BTC Market Hours Daily | xgb | XGBoost | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 148 | 55 | 93 | 37.16% | 37.16% | 37.16% | 12.84 pp | -38 | 13 | -2.92 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
