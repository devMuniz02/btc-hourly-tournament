# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T03:03:45.953679+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 253 | 193 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 288 | 228 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 411 | 216 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 411 | 216 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 187 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 15:00:00+00:00 | 187 | 57 | 130 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 216 | 112 | 104 | 51.85% | 51.85% | 51.85% | 1.85 pp | 8 | 17 | 0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 216 | 104 | 112 | 48.15% | 48.15% | 48.15% | 1.85 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| BTC Market Hours | transformer | Transformer | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 18 | -1.22 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 101 | 117 | 46.33% | 46.33% | 46.33% | 3.67 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 216 | 92 | 124 | 42.59% | 42.59% | 42.59% | 7.41 pp | -32 | 17 | -1.88 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 216 | 90 | 126 | 41.67% | 41.67% | 41.67% | 8.33 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 8 | -2.38 |
| BTC Daily | nn | NN | 218 | 97 | 121 | 44.50% | 44.50% | 44.50% | 5.50 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 8 | -3.62 |
| BTC Market Hours | lstm | LSTM | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 17 | -3.65 |
| BTC Market Hours Daily | lstm | LSTM | 216 | 75 | 141 | 34.72% | 34.72% | 34.72% | 15.28 pp | -66 | 18 | -3.67 |
| BTC Hourly | rf | RandomForest | 193 | 81 | 112 | 41.97% | 41.97% | 41.97% | 8.03 pp | -31 | 8 | -3.88 |
| BTC Daily | transformer | Transformer | 218 | 88 | 130 | 40.37% | 40.37% | 40.37% | 9.63 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 218 | 83 | 135 | 38.07% | 38.07% | 38.07% | 11.93 pp | -52 | 10 | -5.20 |
| BTC Hourly | lstm | LSTM | 193 | 72 | 121 | 37.31% | 37.31% | 37.31% | 12.69 pp | -49 | 8 | -6.12 |
| BTC Daily | xgb | XGBoost | 228 | 80 | 148 | 35.09% | 35.09% | 35.09% | 14.91 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 8 | -6.62 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 193 | 98 | 95 | 50.78% | 50.78% | 50.78% | 0.78 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 193 | 87 | 106 | 45.08% | 45.08% | 45.08% | 4.92 pp | -19 | 8 | -2.38 |
| BTC Hourly | nn | NN | 193 | 82 | 111 | 42.49% | 42.49% | 42.49% | 7.51 pp | -29 | 8 | -3.62 |
| BTC Hourly | rf | RandomForest | 193 | 81 | 112 | 41.97% | 41.97% | 41.97% | 8.03 pp | -31 | 8 | -3.88 |
| BTC Hourly | lstm | LSTM | 193 | 72 | 121 | 37.31% | 37.31% | 37.31% | 12.69 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 193 | 70 | 123 | 36.27% | 36.27% | 36.27% | 13.73 pp | -53 | 8 | -6.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 218 | 101 | 117 | 46.33% | 46.33% | 46.33% | 3.67 pp | -16 | 10 | -1.60 |
| BTC Daily | nn | NN | 218 | 97 | 121 | 44.50% | 44.50% | 44.50% | 5.50 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 218 | 88 | 130 | 40.37% | 40.37% | 40.37% | 9.63 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 218 | 83 | 135 | 38.07% | 38.07% | 38.07% | 11.93 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 228 | 80 | 148 | 35.09% | 35.09% | 35.09% | 14.91 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 218 | 73 | 145 | 33.49% | 33.49% | 33.49% | 16.51 pp | -72 | 10 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 216 | 112 | 104 | 51.85% | 51.85% | 51.85% | 1.85 pp | 8 | 17 | 0.47 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 17 | -0.82 |
| BTC Market Hours | transformer | Transformer | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 216 | 92 | 124 | 42.59% | 42.59% | 42.59% | 7.41 pp | -32 | 17 | -1.88 |
| BTC Market Hours | lstm | LSTM | 216 | 77 | 139 | 35.65% | 35.65% | 35.65% | 14.35 pp | -62 | 17 | -3.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 216 | 105 | 111 | 48.61% | 48.61% | 48.61% | 1.39 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 216 | 104 | 112 | 48.15% | 48.15% | 48.15% | 1.85 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 216 | 102 | 114 | 47.22% | 47.22% | 47.22% | 2.78 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 216 | 97 | 119 | 44.91% | 44.91% | 44.91% | 5.09 pp | -22 | 18 | -1.22 |
| BTC Market Hours Daily | xgb | XGBoost | 216 | 90 | 126 | 41.67% | 41.67% | 41.67% | 8.33 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 216 | 75 | 141 | 34.72% | 34.72% | 34.72% | 15.28 pp | -66 | 18 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 187 | 88 | 99 | 47.06% | 47.06% | 47.06% | 2.94 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 187 | 81 | 106 | 43.32% | 43.32% | 43.32% | 6.68 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 57 | 25 | 32 | 43.86% | 43.86% | 43.86% | 6.14 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
