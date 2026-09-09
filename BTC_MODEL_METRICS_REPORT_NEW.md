# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T08:28:59.560958+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 289 | 229 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 324 | 264 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 473 | 252 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 473 | 252 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 221 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 221 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 75 | 146 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 75 | 146 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 252 | 131 | 121 | 51.98% | 52.08% | 51.98% | 1.98 pp | 10 | 20 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 252 | 125 | 127 | 49.60% | 49.58% | 49.60% | 0.40 pp | -2 | 21 | -0.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 252 | 124 | 128 | 49.21% | 48.75% | 49.21% | 0.79 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 252 | 121 | 131 | 48.02% | 47.50% | 48.02% | 1.98 pp | -10 | 21 | -0.48 |
| Consolidated Hourly | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 252 | 120 | 132 | 47.62% | 47.92% | 47.62% | 2.38 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 252 | 117 | 135 | 46.43% | 45.83% | 46.43% | 3.57 pp | -18 | 20 | -0.90 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 252 | 113 | 139 | 44.84% | 44.58% | 44.84% | 5.16 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 252 | 112 | 140 | 44.44% | 45.00% | 44.44% | 5.56 pp | -28 | 21 | -1.33 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 252 | 110 | 142 | 43.65% | 43.33% | 43.65% | 6.35 pp | -32 | 21 | -1.52 |
| BTC Daily | mlp_sklearn | MLPClassifier | 254 | 118 | 136 | 46.46% | 46.67% | 46.46% | 3.54 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| BTC Market Hours | lstm | LSTM | 252 | 105 | 147 | 41.67% | 42.08% | 41.67% | 8.33 pp | -42 | 20 | -2.10 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Daily | nn | NN | 254 | 115 | 139 | 45.28% | 44.17% | 45.28% | 4.72 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 252 | 103 | 149 | 40.87% | 41.67% | 40.87% | 9.13 pp | -46 | 21 | -2.19 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| BTC Hourly | transformer | Transformer | 229 | 100 | 129 | 43.67% | 43.67% | 43.67% | 6.33 pp | -29 | 10 | -2.90 |
| Consolidated Hourly | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| BTC Hourly | nn | NN | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 10 | -3.90 |
| BTC Hourly | rf | RandomForest | 229 | 94 | 135 | 41.05% | 41.05% | 41.05% | 8.95 pp | -41 | 10 | -4.10 |
| BTC Daily | transformer | Transformer | 254 | 102 | 152 | 40.16% | 39.58% | 40.16% | 9.84 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 254 | 95 | 159 | 37.40% | 36.67% | 37.40% | 12.60 pp | -64 | 11 | -5.82 |
| BTC Hourly | lstm | LSTM | 229 | 85 | 144 | 37.12% | 37.12% | 37.12% | 12.88 pp | -59 | 10 | -5.90 |
| BTC Daily | xgb | XGBoost | 264 | 93 | 171 | 35.23% | 35.00% | 35.23% | 14.77 pp | -78 | 12 | -6.50 |
| BTC Daily | lstm | LSTM | 254 | 88 | 166 | 34.65% | 35.00% | 34.65% | 15.35 pp | -78 | 11 | -7.09 |
| BTC Hourly | xgb | XGBoost | 229 | 79 | 150 | 34.50% | 34.50% | 34.50% | 15.50 pp | -71 | 10 | -7.10 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 229 | 114 | 115 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Hourly | transformer | Transformer | 229 | 100 | 129 | 43.67% | 43.67% | 43.67% | 6.33 pp | -29 | 10 | -2.90 |
| BTC Hourly | nn | NN | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 10 | -3.90 |
| BTC Hourly | rf | RandomForest | 229 | 94 | 135 | 41.05% | 41.05% | 41.05% | 8.95 pp | -41 | 10 | -4.10 |
| BTC Hourly | lstm | LSTM | 229 | 85 | 144 | 37.12% | 37.12% | 37.12% | 12.88 pp | -59 | 10 | -5.90 |
| BTC Hourly | xgb | XGBoost | 229 | 79 | 150 | 34.50% | 34.50% | 34.50% | 15.50 pp | -71 | 10 | -7.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 254 | 118 | 136 | 46.46% | 46.67% | 46.46% | 3.54 pp | -18 | 11 | -1.64 |
| BTC Daily | nn | NN | 254 | 115 | 139 | 45.28% | 44.17% | 45.28% | 4.72 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 254 | 102 | 152 | 40.16% | 39.58% | 40.16% | 9.84 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 254 | 95 | 159 | 37.40% | 36.67% | 37.40% | 12.60 pp | -64 | 11 | -5.82 |
| BTC Daily | xgb | XGBoost | 264 | 93 | 171 | 35.23% | 35.00% | 35.23% | 14.77 pp | -78 | 12 | -6.50 |
| BTC Daily | lstm | LSTM | 254 | 88 | 166 | 34.65% | 35.00% | 34.65% | 15.35 pp | -78 | 11 | -7.09 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 252 | 131 | 121 | 51.98% | 52.08% | 51.98% | 1.98 pp | 10 | 20 | 0.50 |
| BTC Market Hours | transformer | Transformer | 252 | 120 | 132 | 47.62% | 47.92% | 47.62% | 2.38 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 252 | 118 | 134 | 46.83% | 46.67% | 46.83% | 3.17 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 252 | 117 | 135 | 46.43% | 45.83% | 46.43% | 3.57 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 252 | 113 | 139 | 44.84% | 44.58% | 44.84% | 5.16 pp | -26 | 20 | -1.30 |
| BTC Market Hours | lstm | LSTM | 252 | 105 | 147 | 41.67% | 42.08% | 41.67% | 8.33 pp | -42 | 20 | -2.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 252 | 125 | 127 | 49.60% | 49.58% | 49.60% | 0.40 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 252 | 124 | 128 | 49.21% | 48.75% | 49.21% | 0.79 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 252 | 121 | 131 | 48.02% | 47.50% | 48.02% | 1.98 pp | -10 | 21 | -0.48 |
| BTC Market Hours Daily | xgb | XGBoost | 252 | 112 | 140 | 44.44% | 45.00% | 44.44% | 5.56 pp | -28 | 21 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 252 | 110 | 142 | 43.65% | 43.33% | 43.65% | 6.35 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 252 | 103 | 149 | 40.87% | 41.67% | 40.87% | 9.13 pp | -46 | 21 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
