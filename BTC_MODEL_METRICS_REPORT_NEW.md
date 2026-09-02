# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T11:09:59.448493+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 177 | 117 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 213 | 153 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 271 | 141 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 271 | 141 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 117 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T16:00:00+00:00 | 118 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 141 | 73 | 68 | 51.77% | 51.77% | 51.77% | 1.77 pp | 5 | 11 | 0.45 |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 141 | 68 | 73 | 48.23% | 48.23% | 48.23% | 1.77 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 12 | -0.75 |
| BTC Market Hours | rf | RandomForest | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| BTC Hourly | transformer | Transformer | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 5 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| BTC Market Hours | transformer | Transformer | 141 | 64 | 77 | 45.39% | 45.39% | 45.39% | 4.61 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | nn | NN | 141 | 62 | 79 | 43.97% | 43.97% | 43.97% | 6.03 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 141 | 62 | 79 | 43.97% | 43.97% | 43.97% | 6.03 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 141 | 57 | 84 | 40.43% | 40.43% | 40.43% | 9.57 pp | -27 | 12 | -2.25 |
| BTC Daily | transformer | Transformer | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 141 | 56 | 85 | 39.72% | 39.72% | 39.72% | 10.28 pp | -29 | 11 | -2.64 |
| BTC Market Hours Daily | lstm | LSTM | 141 | 53 | 88 | 37.59% | 37.59% | 37.59% | 12.41 pp | -35 | 12 | -2.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| BTC Daily | xgb | XGBoost | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 8 | -5.38 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |
| BTC Daily | lstm | LSTM | 143 | 50 | 93 | 34.97% | 34.97% | 34.97% | 15.03 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 117 | 38 | 79 | 32.48% | 32.48% | 32.48% | 17.52 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 5 | -0.20 |
| BTC Hourly | transformer | Transformer | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 5 | -1.80 |
| BTC Hourly | rf | RandomForest | 117 | 49 | 68 | 41.88% | 41.88% | 41.88% | 8.12 pp | -19 | 5 | -3.80 |
| BTC Hourly | xgb | XGBoost | 117 | 43 | 74 | 36.75% | 36.75% | 36.75% | 13.25 pp | -31 | 5 | -6.20 |
| BTC Hourly | lstm | LSTM | 117 | 38 | 79 | 32.48% | 32.48% | 32.48% | 17.52 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 7 | -0.71 |
| BTC Daily | nn | NN | 143 | 64 | 79 | 44.76% | 44.76% | 44.76% | 5.24 pp | -15 | 7 | -2.14 |
| BTC Daily | transformer | Transformer | 143 | 63 | 80 | 44.06% | 44.06% | 44.06% | 5.94 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 143 | 59 | 84 | 41.26% | 41.26% | 41.26% | 8.74 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 153 | 55 | 98 | 35.95% | 35.95% | 35.95% | 14.05 pp | -43 | 8 | -5.38 |
| BTC Daily | lstm | LSTM | 143 | 50 | 93 | 34.97% | 34.97% | 34.97% | 15.03 pp | -43 | 7 | -6.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 141 | 73 | 68 | 51.77% | 51.77% | 51.77% | 1.77 pp | 5 | 11 | 0.45 |
| BTC Market Hours | rf | RandomForest | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Market Hours | transformer | Transformer | 141 | 64 | 77 | 45.39% | 45.39% | 45.39% | 4.61 pp | -13 | 11 | -1.18 |
| BTC Market Hours | xgb | XGBoost | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 141 | 56 | 85 | 39.72% | 39.72% | 39.72% | 10.28 pp | -29 | 11 | -2.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 141 | 68 | 73 | 48.23% | 48.23% | 48.23% | 1.77 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | transformer | Transformer | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 12 | -0.75 |
| BTC Market Hours Daily | nn | NN | 141 | 62 | 79 | 43.97% | 43.97% | 43.97% | 6.03 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 141 | 62 | 79 | 43.97% | 43.97% | 43.97% | 6.03 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | xgb | XGBoost | 141 | 57 | 84 | 40.43% | 40.43% | 40.43% | 9.57 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 141 | 53 | 88 | 37.59% | 37.59% | 37.59% | 12.41 pp | -35 | 12 | -2.92 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 117 | 58 | 59 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 117 | 56 | 61 | 47.86% | 47.86% | 47.86% | 2.14 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 117 | 54 | 63 | 46.15% | 46.15% | 46.15% | 3.85 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 117 | 51 | 66 | 43.59% | 43.59% | 43.59% | 6.41 pp | -15 | 10 | -1.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 19 | 7 | 12 | 36.84% | 36.84% | 36.84% | 13.16 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 19 | 4 | 15 | 21.05% | 21.05% | 21.05% | 28.95 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 4 | 16 | 20.00% | 20.00% | 20.00% | 30.00 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
