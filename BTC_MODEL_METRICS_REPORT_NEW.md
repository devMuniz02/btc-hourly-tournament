# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T10:15:16.030720+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 208 | 148 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 244 | 184 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 328 | 172 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 328 | 172 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 147 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 147 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 35 | 112 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 19:00:00+00:00 | 147 | 35 | 112 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 7 | 0.86 |
| BTC Market Hours | nn | NN | 172 | 90 | 82 | 52.33% | 52.33% | 52.33% | 2.33 pp | 8 | 14 | 0.57 |
| Consolidated Hourly | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Market Hours | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 172 | 85 | 87 | 49.42% | 49.42% | 49.42% | 0.58 pp | -2 | 15 | -0.13 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 15 | -0.27 |
| BTC Hourly | transformer | Transformer | 148 | 72 | 76 | 48.65% | 48.65% | 48.65% | 1.35 pp | -4 | 7 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| BTC Market Hours Daily | nn | NN | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 15 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | rf | RandomForest | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 15 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 14 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 15 | -2.00 |
| BTC Market Hours | lstm | LSTM | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 14 | -2.14 |
| Consolidated Hourly | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |
| BTC Daily | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 172 | 67 | 105 | 38.95% | 38.95% | 38.95% | 11.05 pp | -38 | 15 | -2.53 |
| BTC Hourly | nn | NN | 148 | 64 | 84 | 43.24% | 43.24% | 43.24% | 6.76 pp | -20 | 7 | -2.86 |
| BTC Daily | transformer | Transformer | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 174 | 71 | 103 | 40.80% | 40.80% | 40.80% | 9.20 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 9 | -5.33 |
| BTC Hourly | lstm | LSTM | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 148 | 53 | 95 | 35.81% | 35.81% | 35.81% | 14.19 pp | -42 | 7 | -6.00 |
| BTC Daily | lstm | LSTM | 174 | 60 | 114 | 34.48% | 34.48% | 34.48% | 15.52 pp | -54 | 8 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 148 | 72 | 76 | 48.65% | 48.65% | 48.65% | 1.35 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 148 | 64 | 84 | 43.24% | 43.24% | 43.24% | 6.76 pp | -20 | 7 | -2.86 |
| BTC Hourly | rf | RandomForest | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 148 | 53 | 95 | 35.81% | 35.81% | 35.81% | 14.19 pp | -42 | 7 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 174 | 71 | 103 | 40.80% | 40.80% | 40.80% | 9.20 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 174 | 60 | 114 | 34.48% | 34.48% | 34.48% | 15.52 pp | -54 | 8 | -6.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 172 | 90 | 82 | 52.33% | 52.33% | 52.33% | 2.33 pp | 8 | 14 | 0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | rf | RandomForest | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 172 | 85 | 87 | 49.42% | 49.42% | 49.42% | 0.58 pp | -2 | 15 | -0.13 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 15 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 15 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 172 | 67 | 105 | 38.95% | 38.95% | 38.95% | 11.05 pp | -38 | 15 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 147 | 76 | 71 | 51.70% | 51.70% | 51.70% | 1.70 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 147 | 70 | 77 | 47.62% | 47.62% | 47.62% | 2.38 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 147 | 68 | 79 | 46.26% | 46.26% | 46.26% | 3.74 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 147 | 67 | 80 | 45.58% | 45.58% | 45.58% | 4.42 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 147 | 63 | 84 | 42.86% | 42.86% | 42.86% | 7.14 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 147 | 61 | 86 | 41.50% | 41.50% | 41.50% | 8.50 pp | -25 | 11 | -2.27 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 3 | -4.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
