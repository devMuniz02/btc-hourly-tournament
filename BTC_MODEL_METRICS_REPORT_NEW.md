# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T00:07:50.795040+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 186 | 126 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 221 | 161 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 23:00:00+00:00 | 291 | 149 | 142 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 23:00:00+00:00 | 291 | 149 | 142 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| BTC Market Hours | nn | NN | 149 | 78 | 71 | 52.35% | 52.35% | 52.35% | 2.35 pp | 7 | 12 | 0.58 |
| BTC Hourly | transformer | Transformer | 126 | 63 | 63 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 149 | 71 | 78 | 47.65% | 47.65% | 47.65% | 2.35 pp | -7 | 13 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 7 | -0.71 |
| BTC Market Hours | rf | RandomForest | 149 | 69 | 80 | 46.31% | 46.31% | 46.31% | 3.69 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 149 | 67 | 82 | 44.97% | 44.97% | 44.97% | 5.03 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 13 | -1.31 |
| BTC Market Hours | transformer | Transformer | 149 | 66 | 83 | 44.30% | 44.30% | 44.30% | 5.70 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 13 | -2.08 |
| BTC Market Hours | xgb | XGBoost | 149 | 62 | 87 | 41.61% | 41.61% | 41.61% | 8.39 pp | -25 | 12 | -2.08 |
| BTC Market Hours | lstm | LSTM | 149 | 60 | 89 | 40.27% | 40.27% | 40.27% | 9.73 pp | -29 | 12 | -2.42 |
| Consolidated Hourly | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| BTC Daily | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 7 | -2.71 |
| BTC Daily | transformer | Transformer | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 149 | 56 | 93 | 37.58% | 37.58% | 37.58% | 12.42 pp | -37 | 13 | -2.85 |
| BTC Hourly | rf | RandomForest | 126 | 52 | 74 | 41.27% | 41.27% | 41.27% | 8.73 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 7 | -3.86 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 6 | -5.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 161 | 57 | 104 | 35.40% | 35.40% | 35.40% | 14.60 pp | -47 | 8 | -5.88 |
| BTC Daily | lstm | LSTM | 151 | 53 | 98 | 35.10% | 35.10% | 35.10% | 14.90 pp | -45 | 7 | -6.43 |
| BTC Hourly | lstm | LSTM | 126 | 43 | 83 | 34.13% | 34.13% | 34.13% | 15.87 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 126 | 63 | 63 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 6 | -1.67 |
| BTC Hourly | rf | RandomForest | 126 | 52 | 74 | 41.27% | 41.27% | 41.27% | 8.73 pp | -22 | 6 | -3.67 |
| BTC Hourly | xgb | XGBoost | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 126 | 43 | 83 | 34.13% | 34.13% | 34.13% | 15.87 pp | -40 | 6 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 151 | 73 | 78 | 48.34% | 48.34% | 48.34% | 1.66 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 7 | -2.71 |
| BTC Daily | transformer | Transformer | 151 | 66 | 85 | 43.71% | 43.71% | 43.71% | 6.29 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 151 | 62 | 89 | 41.06% | 41.06% | 41.06% | 8.94 pp | -27 | 7 | -3.86 |
| BTC Daily | xgb | XGBoost | 161 | 57 | 104 | 35.40% | 35.40% | 35.40% | 14.60 pp | -47 | 8 | -5.88 |
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
