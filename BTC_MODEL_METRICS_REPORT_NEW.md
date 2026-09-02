# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T23:09:56.802457+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 185 | 125 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 221 | 161 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 22:00:00+00:00 | 290 | 149 | 141 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 22:00:00+00:00 | 290 | 149 | 141 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 125 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 126 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 149 | 78 | 71 | 52.35% | 52.35% | 52.35% | 2.35 pp | 7 | 12 | 0.58 |
| Consolidated Hourly | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | transformer | Transformer | 125 | 62 | 63 | 49.60% | 49.60% | 49.60% | 0.40 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 7 | -0.43 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| BTC Market Hours | rf | RandomForest | 149 | 69 | 80 | 46.31% | 46.31% | 46.31% | 3.69 pp | -11 | 12 | -0.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | nn | NN | 149 | 67 | 82 | 44.97% | 44.97% | 44.97% | 5.03 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 13 | -1.31 |
| BTC Market Hours | transformer | Transformer | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |
| BTC Hourly | nn | NN | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 13 | -2.08 |
| BTC Market Hours | xgb | XGBoost | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 149 | 60 | 89 | 40.27% | 40.27% | 40.27% | 9.73 pp | -29 | 12 | -2.42 |
| BTC Daily | nn | NN | 151 | 67 | 84 | 44.37% | 44.37% | 44.37% | 5.63 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 151 | 67 | 84 | 44.37% | 44.37% | 44.37% | 5.63 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | lstm | LSTM | 149 | 56 | 93 | 37.58% | 37.58% | 37.58% | 12.42 pp | -37 | 13 | -2.85 |
| BTC Daily | rf | RandomForest | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 125 | 51 | 74 | 40.80% | 40.80% | 40.80% | 9.20 pp | -23 | 6 | -3.83 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 125 | 46 | 79 | 36.80% | 36.80% | 36.80% | 13.20 pp | -33 | 6 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 161 | 58 | 103 | 36.02% | 36.02% | 36.02% | 13.98 pp | -45 | 8 | -5.62 |
| BTC Daily | lstm | LSTM | 151 | 53 | 98 | 35.10% | 35.10% | 35.10% | 14.90 pp | -45 | 7 | -6.43 |
| BTC Hourly | lstm | LSTM | 125 | 42 | 83 | 33.60% | 33.60% | 33.60% | 16.40 pp | -41 | 6 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 125 | 62 | 63 | 49.60% | 49.60% | 49.60% | 0.40 pp | -1 | 6 | -0.17 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 6 | -1.83 |
| BTC Hourly | rf | RandomForest | 125 | 51 | 74 | 40.80% | 40.80% | 40.80% | 9.20 pp | -23 | 6 | -3.83 |
| BTC Hourly | xgb | XGBoost | 125 | 46 | 79 | 36.80% | 36.80% | 36.80% | 13.20 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 125 | 42 | 83 | 33.60% | 33.60% | 33.60% | 16.40 pp | -41 | 6 | -6.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 151 | 74 | 77 | 49.01% | 49.01% | 49.01% | 0.99 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 151 | 67 | 84 | 44.37% | 44.37% | 44.37% | 5.63 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 151 | 67 | 84 | 44.37% | 44.37% | 44.37% | 5.63 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 151 | 63 | 88 | 41.72% | 41.72% | 41.72% | 8.28 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 161 | 58 | 103 | 36.02% | 36.02% | 36.02% | 13.98 pp | -45 | 8 | -5.62 |
| BTC Daily | lstm | LSTM | 151 | 53 | 98 | 35.10% | 35.10% | 35.10% | 14.90 pp | -45 | 7 | -6.43 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 149 | 78 | 71 | 52.35% | 52.35% | 52.35% | 2.35 pp | 7 | 12 | 0.58 |
| BTC Market Hours | rf | RandomForest | 149 | 69 | 80 | 46.31% | 46.31% | 46.31% | 3.69 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 12 | -1.08 |
| BTC Market Hours | transformer | Transformer | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 12 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 149 | 60 | 89 | 40.27% | 40.27% | 40.27% | 9.73 pp | -29 | 12 | -2.42 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | nn | NN | 149 | 67 | 82 | 44.97% | 44.97% | 44.97% | 5.03 pp | -15 | 13 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 149 | 56 | 93 | 37.58% | 37.58% | 37.58% | 12.42 pp | -37 | 13 | -2.85 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 64 | 61 | 51.20% | 51.20% | 51.20% | 1.20 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 58 | 67 | 46.40% | 46.40% | 46.40% | 3.60 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 55 | 70 | 44.00% | 44.00% | 44.00% | 6.00 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 54 | 71 | 43.20% | 43.20% | 43.20% | 6.80 pp | -17 | 10 | -1.70 |

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
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
