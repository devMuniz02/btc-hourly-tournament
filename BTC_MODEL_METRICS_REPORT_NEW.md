# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T21:33:43.277893+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 20:00:00+00:00 | 287 | 148 | 139 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 20:00:00+00:00 | 287 | 148 | 139 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 123 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 123 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 22 | 101 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 22 | 101 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| BTC Market Hours | nn | NN | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 12 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Hourly | transformer | Transformer | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 148 | 71 | 77 | 47.97% | 47.97% | 47.97% | 2.03 pp | -6 | 13 | -0.46 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 150 | 73 | 77 | 48.67% | 48.67% | 48.67% | 1.33 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 148 | 70 | 78 | 47.30% | 47.30% | 47.30% | 2.70 pp | -8 | 13 | -0.62 |
| Consolidated Hourly | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| BTC Market Hours | rf | RandomForest | 148 | 69 | 79 | 46.62% | 46.62% | 46.62% | 3.38 pp | -10 | 12 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 148 | 67 | 81 | 45.27% | 45.27% | 45.27% | 4.73 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 148 | 66 | 82 | 44.59% | 44.59% | 44.59% | 5.41 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| BTC Market Hours | transformer | Transformer | 148 | 65 | 83 | 43.92% | 43.92% | 43.92% | 6.08 pp | -18 | 12 | -1.50 |
| BTC Hourly | nn | NN | 124 | 57 | 67 | 45.97% | 45.97% | 45.97% | 4.03 pp | -10 | 6 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 148 | 62 | 86 | 41.89% | 41.89% | 41.89% | 8.11 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Daily | nn | NN | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| BTC Daily | transformer | Transformer | 150 | 67 | 83 | 44.67% | 44.67% | 44.67% | 5.33 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |
| BTC Market Hours | lstm | LSTM | 148 | 59 | 89 | 39.86% | 39.86% | 39.86% | 10.14 pp | -30 | 12 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 148 | 55 | 93 | 37.16% | 37.16% | 37.16% | 12.84 pp | -38 | 13 | -2.92 |
| BTC Daily | rf | RandomForest | 150 | 63 | 87 | 42.00% | 42.00% | 42.00% | 8.00 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 124 | 51 | 73 | 41.13% | 41.13% | 41.13% | 8.87 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| BTC Hourly | xgb | XGBoost | 124 | 46 | 78 | 37.10% | 37.10% | 37.10% | 12.90 pp | -32 | 6 | -5.33 |
| BTC Daily | xgb | XGBoost | 160 | 58 | 102 | 36.25% | 36.25% | 36.25% | 13.75 pp | -44 | 8 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |
| BTC Daily | lstm | LSTM | 150 | 53 | 97 | 35.33% | 35.33% | 35.33% | 14.67 pp | -44 | 7 | -6.29 |
| BTC Hourly | lstm | LSTM | 124 | 42 | 82 | 33.87% | 33.87% | 33.87% | 16.13 pp | -40 | 6 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
| BTC Hourly | transformer | Transformer | 124 | 61 | 63 | 49.19% | 49.19% | 49.19% | 0.81 pp | -2 | 6 | -0.33 |
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
| Consolidated Hourly | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
