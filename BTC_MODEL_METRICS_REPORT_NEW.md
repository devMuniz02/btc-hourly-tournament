# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T00:39:28.566690+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 222 | 162 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 23:00:00+00:00 | 292 | 150 | 142 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 23:00:00+00:00 | 292 | 150 | 142 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 126 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 126 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 126 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T20:00:00+00:00 | 127 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 150 | 78 | 72 | 52.00% | 52.00% | 52.00% | 2.00 pp | 6 | 12 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 126 | 64 | 62 | 50.79% | 50.79% | 50.79% | 0.79 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 126 | 64 | 62 | 50.79% | 50.79% | 50.79% | 0.79 pp | 2 | 10 | 0.20 |
| BTC Hourly | transformer | Transformer | 126 | 63 | 63 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 10 | -0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 126 | 60 | 66 | 47.62% | 47.62% | 47.62% | 2.38 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 126 | 60 | 66 | 47.62% | 47.62% | 47.62% | 2.38 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 10 | -1.00 |
| BTC Market Hours | rf | RandomForest | 150 | 69 | 81 | 46.00% | 46.00% | 46.00% | 4.00 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 12 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 150 | 66 | 84 | 44.00% | 44.00% | 44.00% | 6.00 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| BTC Hourly | nn | NN | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 150 | 63 | 87 | 42.00% | 42.00% | 42.00% | 8.00 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 13 | -2.00 |
| BTC Market Hours | lstm | LSTM | 150 | 61 | 89 | 40.67% | 40.67% | 40.67% | 9.33 pp | -28 | 12 | -2.33 |
| BTC Daily | nn | NN | 152 | 67 | 85 | 44.08% | 44.08% | 44.08% | 5.92 pp | -18 | 7 | -2.57 |
| BTC Daily | transformer | Transformer | 152 | 67 | 85 | 44.08% | 44.08% | 44.08% | 5.92 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 150 | 56 | 94 | 37.33% | 37.33% | 37.33% | 12.67 pp | -38 | 13 | -2.92 |
| BTC Hourly | rf | RandomForest | 126 | 52 | 74 | 41.27% | 41.27% | 41.27% | 8.73 pp | -22 | 6 | -3.67 |
| BTC Daily | rf | RandomForest | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 126 | 47 | 79 | 37.30% | 37.30% | 37.30% | 12.70 pp | -32 | 6 | -5.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 162 | 58 | 104 | 35.80% | 35.80% | 35.80% | 14.20 pp | -46 | 8 | -5.75 |
| BTC Daily | lstm | LSTM | 152 | 53 | 99 | 34.87% | 34.87% | 34.87% | 15.13 pp | -46 | 7 | -6.57 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 152 | 67 | 85 | 44.08% | 44.08% | 44.08% | 5.92 pp | -18 | 7 | -2.57 |
| BTC Daily | transformer | Transformer | 152 | 67 | 85 | 44.08% | 44.08% | 44.08% | 5.92 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 152 | 63 | 89 | 41.45% | 41.45% | 41.45% | 8.55 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 162 | 58 | 104 | 35.80% | 35.80% | 35.80% | 14.20 pp | -46 | 8 | -5.75 |
| BTC Daily | lstm | LSTM | 152 | 53 | 99 | 34.87% | 34.87% | 34.87% | 15.13 pp | -46 | 7 | -6.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 150 | 78 | 72 | 52.00% | 52.00% | 52.00% | 2.00 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 150 | 69 | 81 | 46.00% | 46.00% | 46.00% | 4.00 pp | -12 | 12 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 12 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 150 | 63 | 87 | 42.00% | 42.00% | 42.00% | 8.00 pp | -24 | 12 | -2.00 |
| BTC Market Hours | lstm | LSTM | 150 | 61 | 89 | 40.67% | 40.67% | 40.67% | 9.33 pp | -28 | 12 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 13 | -0.46 |
| BTC Market Hours Daily | nn | NN | 150 | 68 | 82 | 45.33% | 45.33% | 45.33% | 4.67 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 150 | 66 | 84 | 44.00% | 44.00% | 44.00% | 6.00 pp | -18 | 13 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 150 | 62 | 88 | 41.33% | 41.33% | 41.33% | 8.67 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 150 | 56 | 94 | 37.33% | 37.33% | 37.33% | 12.67 pp | -38 | 13 | -2.92 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 126 | 64 | 62 | 50.79% | 50.79% | 50.79% | 0.79 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 10 | -0.20 |
| Consolidated Hourly | xgb | XGBoost | 126 | 60 | 66 | 47.62% | 47.62% | 47.62% | 2.38 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 126 | 64 | 62 | 50.79% | 50.79% | 50.79% | 0.79 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 126 | 60 | 66 | 47.62% | 47.62% | 47.62% | 2.38 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 126 | 58 | 68 | 46.03% | 46.03% | 46.03% | 3.97 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 126 | 55 | 71 | 43.65% | 43.65% | 43.65% | 6.35 pp | -16 | 10 | -1.60 |

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
