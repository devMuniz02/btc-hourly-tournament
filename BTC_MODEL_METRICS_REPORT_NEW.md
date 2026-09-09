# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T20:27:03.385716+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 296 | 236 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 332 | 272 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 19:00:00+00:00 | 489 | 260 | 229 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 19:00:00+00:00 | 489 | 260 | 229 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 227 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T00:00:00+00:00 | 228 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 260 | 137 | 123 | 52.69% | 52.50% | 52.69% | 2.69 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 260 | 130 | 130 | 50.00% | 49.58% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 260 | 128 | 132 | 49.23% | 48.75% | 49.23% | 0.77 pp | -4 | 21 | -0.19 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | nn | NN | 260 | 125 | 135 | 48.08% | 48.33% | 48.08% | 1.92 pp | -10 | 21 | -0.48 |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 260 | 123 | 137 | 47.31% | 47.92% | 47.31% | 2.69 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 260 | 122 | 138 | 46.92% | 47.08% | 46.92% | 3.08 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 260 | 120 | 140 | 46.15% | 44.17% | 46.15% | 3.85 pp | -20 | 20 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 260 | 116 | 144 | 44.62% | 43.33% | 44.62% | 5.38 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 260 | 115 | 145 | 44.23% | 43.33% | 44.23% | 5.77 pp | -30 | 21 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 262 | 122 | 140 | 46.56% | 45.83% | 46.56% | 3.44 pp | -18 | 12 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 260 | 113 | 147 | 43.46% | 42.50% | 43.46% | 6.54 pp | -34 | 21 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Daily | nn | NN | 262 | 120 | 142 | 45.80% | 45.00% | 45.80% | 4.20 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 260 | 108 | 152 | 41.54% | 42.08% | 41.54% | 8.46 pp | -44 | 21 | -2.10 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| BTC Market Hours | lstm | LSTM | 260 | 106 | 154 | 40.77% | 42.08% | 40.77% | 9.23 pp | -48 | 20 | -2.40 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Hourly | transformer | Transformer | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 236 | 99 | 137 | 41.95% | 41.95% | 41.95% | 8.05 pp | -38 | 10 | -3.80 |
| BTC Daily | transformer | Transformer | 262 | 107 | 155 | 40.84% | 40.00% | 40.84% | 9.16 pp | -48 | 12 | -4.00 |
| BTC Hourly | rf | RandomForest | 236 | 97 | 139 | 41.10% | 41.10% | 41.10% | 8.90 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 262 | 98 | 164 | 37.40% | 37.08% | 37.40% | 12.60 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 272 | 99 | 173 | 36.40% | 36.25% | 36.40% | 13.60 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 236 | 87 | 149 | 36.86% | 36.86% | 36.86% | 13.14 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 262 | 92 | 170 | 35.11% | 35.83% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 236 | 81 | 155 | 34.32% | 34.32% | 34.32% | 15.68 pp | -74 | 10 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 236 | 116 | 120 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 236 | 105 | 131 | 44.49% | 44.49% | 44.49% | 5.51 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 236 | 99 | 137 | 41.95% | 41.95% | 41.95% | 8.05 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 236 | 97 | 139 | 41.10% | 41.10% | 41.10% | 8.90 pp | -42 | 10 | -4.20 |
| BTC Hourly | lstm | LSTM | 236 | 87 | 149 | 36.86% | 36.86% | 36.86% | 13.14 pp | -62 | 10 | -6.20 |
| BTC Hourly | xgb | XGBoost | 236 | 81 | 155 | 34.32% | 34.32% | 34.32% | 15.68 pp | -74 | 10 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 262 | 122 | 140 | 46.56% | 45.83% | 46.56% | 3.44 pp | -18 | 12 | -1.50 |
| BTC Daily | nn | NN | 262 | 120 | 142 | 45.80% | 45.00% | 45.80% | 4.20 pp | -22 | 12 | -1.83 |
| BTC Daily | transformer | Transformer | 262 | 107 | 155 | 40.84% | 40.00% | 40.84% | 9.16 pp | -48 | 12 | -4.00 |
| BTC Daily | rf | RandomForest | 262 | 98 | 164 | 37.40% | 37.08% | 37.40% | 12.60 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 272 | 99 | 173 | 36.40% | 36.25% | 36.40% | 13.60 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 262 | 92 | 170 | 35.11% | 35.83% | 35.11% | 14.89 pp | -78 | 12 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 260 | 137 | 123 | 52.69% | 52.50% | 52.69% | 2.69 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 260 | 123 | 137 | 47.31% | 47.92% | 47.31% | 2.69 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 260 | 122 | 138 | 46.92% | 47.08% | 46.92% | 3.08 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 260 | 120 | 140 | 46.15% | 44.17% | 46.15% | 3.85 pp | -20 | 20 | -1.00 |
| BTC Market Hours | rf | RandomForest | 260 | 116 | 144 | 44.62% | 43.33% | 44.62% | 5.38 pp | -28 | 20 | -1.40 |
| BTC Market Hours | lstm | LSTM | 260 | 106 | 154 | 40.77% | 42.08% | 40.77% | 9.23 pp | -48 | 20 | -2.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 260 | 130 | 130 | 50.00% | 49.58% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 260 | 128 | 132 | 49.23% | 48.75% | 49.23% | 0.77 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 260 | 125 | 135 | 48.08% | 48.33% | 48.08% | 1.92 pp | -10 | 21 | -0.48 |
| BTC Market Hours Daily | xgb | XGBoost | 260 | 115 | 145 | 44.23% | 43.33% | 44.23% | 5.77 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 260 | 113 | 147 | 43.46% | 42.50% | 43.46% | 6.54 pp | -34 | 21 | -1.62 |
| BTC Market Hours Daily | lstm | LSTM | 260 | 108 | 152 | 41.54% | 42.08% | 41.54% | 8.46 pp | -44 | 21 | -2.10 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 227 | 109 | 118 | 48.02% | 48.02% | 48.02% | 1.98 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 227 | 105 | 122 | 46.26% | 46.26% | 46.26% | 3.74 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 227 | 103 | 124 | 45.37% | 45.37% | 45.37% | 4.63 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 227 | 95 | 132 | 41.85% | 41.85% | 41.85% | 8.15 pp | -37 | 15 | -2.47 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | nn | NN | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
