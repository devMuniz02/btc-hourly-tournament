# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T12:34:35.437149+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 210 | 150 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 246 | 186 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 330 | 174 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 330 | 174 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 148 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 148 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 148 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T19:00:00+00:00 | 149 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 150 | 78 | 72 | 52.00% | 52.00% | 52.00% | 2.00 pp | 6 | 7 | 0.86 |
| BTC Market Hours | nn | NN | 174 | 91 | 83 | 52.30% | 52.30% | 52.30% | 2.30 pp | 8 | 14 | 0.57 |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 174 | 87 | 87 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 15 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 148 | 74 | 74 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 148 | 74 | 74 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 15 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| BTC Market Hours | transformer | Transformer | 174 | 82 | 92 | 47.13% | 47.13% | 47.13% | 2.87 pp | -10 | 14 | -0.71 |
| BTC Market Hours Daily | nn | NN | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 15 | -0.80 |
| BTC Hourly | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 7 | -0.86 |
| BTC Market Hours | rf | RandomForest | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 14 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 174 | 78 | 96 | 44.83% | 44.83% | 44.83% | 5.17 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | nn | NN | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 14 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 15 | -1.87 |
| BTC Market Hours | lstm | LSTM | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 11 | -2.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 174 | 69 | 105 | 39.66% | 39.66% | 39.66% | 10.34 pp | -36 | 15 | -2.40 |
| BTC Daily | nn | NN | 176 | 78 | 98 | 44.32% | 44.32% | 44.32% | 5.68 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 176 | 76 | 100 | 43.18% | 43.18% | 43.18% | 6.82 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| BTC Hourly | nn | NN | 150 | 64 | 86 | 42.67% | 42.67% | 42.67% | 7.33 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 176 | 72 | 104 | 40.91% | 40.91% | 40.91% | 9.09 pp | -32 | 8 | -4.00 |
| BTC Hourly | rf | RandomForest | 150 | 61 | 89 | 40.67% | 40.67% | 40.67% | 9.33 pp | -28 | 7 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 3 | -4.67 |
| BTC Daily | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 9 | -5.33 |
| BTC Hourly | lstm | LSTM | 150 | 54 | 96 | 36.00% | 36.00% | 36.00% | 14.00 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 150 | 53 | 97 | 35.33% | 35.33% | 35.33% | 14.67 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 176 | 62 | 114 | 35.23% | 35.23% | 35.23% | 14.77 pp | -52 | 8 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 150 | 78 | 72 | 52.00% | 52.00% | 52.00% | 2.00 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 150 | 72 | 78 | 48.00% | 48.00% | 48.00% | 2.00 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 150 | 64 | 86 | 42.67% | 42.67% | 42.67% | 7.33 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 150 | 61 | 89 | 40.67% | 40.67% | 40.67% | 9.33 pp | -28 | 7 | -4.00 |
| BTC Hourly | lstm | LSTM | 150 | 54 | 96 | 36.00% | 36.00% | 36.00% | 14.00 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 150 | 53 | 97 | 35.33% | 35.33% | 35.33% | 14.67 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 176 | 81 | 95 | 46.02% | 46.02% | 46.02% | 3.98 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 176 | 78 | 98 | 44.32% | 44.32% | 44.32% | 5.68 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 176 | 76 | 100 | 43.18% | 43.18% | 43.18% | 6.82 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 176 | 72 | 104 | 40.91% | 40.91% | 40.91% | 9.09 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 176 | 62 | 114 | 35.23% | 35.23% | 35.23% | 14.77 pp | -52 | 8 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 174 | 91 | 83 | 52.30% | 52.30% | 52.30% | 2.30 pp | 8 | 14 | 0.57 |
| BTC Market Hours | transformer | Transformer | 174 | 82 | 92 | 47.13% | 47.13% | 47.13% | 2.87 pp | -10 | 14 | -0.71 |
| BTC Market Hours | rf | RandomForest | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 14 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 14 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 14 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 174 | 87 | 87 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 15 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 15 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 174 | 78 | 96 | 44.83% | 44.83% | 44.83% | 5.17 pp | -18 | 15 | -1.20 |
| BTC Market Hours Daily | xgb | XGBoost | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | lstm | LSTM | 174 | 69 | 105 | 39.66% | 39.66% | 39.66% | 10.34 pp | -36 | 15 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 148 | 74 | 74 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | lstm | LSTM | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 11 | -1.64 |
| Consolidated Hourly | transformer | Transformer | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 11 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 148 | 74 | 74 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 11 | -0.55 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 11 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 18 | 18 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
