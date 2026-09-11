# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T02:38:21.508477+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 316 | 256 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 352 | 292 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 527 | 280 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 527 | 280 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 246 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 246 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 246 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 247 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 280 | 142 | 138 | 50.71% | 50.00% | 50.71% | 0.71 pp | 4 | 22 | 0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 280 | 138 | 142 | 49.29% | 48.75% | 49.29% | 0.71 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 280 | 135 | 145 | 48.21% | 48.33% | 48.21% | 1.79 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | nn | NN | 280 | 134 | 146 | 47.86% | 48.75% | 47.86% | 2.14 pp | -12 | 23 | -0.52 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 256 | 123 | 133 | 48.05% | 47.92% | 48.05% | 1.95 pp | -10 | 11 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 280 | 129 | 151 | 46.07% | 46.25% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 280 | 129 | 151 | 46.07% | 45.42% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 246 | 115 | 131 | 46.75% | 46.67% | 46.75% | 3.25 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 246 | 115 | 131 | 46.75% | 46.67% | 46.75% | 3.25 pp | -16 | 15 | -1.07 |
| BTC Market Hours | xgb | XGBoost | 280 | 126 | 154 | 45.00% | 45.00% | 45.00% | 5.00 pp | -28 | 22 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 246 | 113 | 133 | 45.93% | 46.25% | 45.93% | 4.07 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 246 | 113 | 133 | 45.93% | 46.25% | 45.93% | 4.07 pp | -20 | 15 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 280 | 124 | 156 | 44.29% | 42.92% | 44.29% | 5.71 pp | -32 | 22 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 15 | -1.47 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 280 | 121 | 159 | 43.21% | 42.50% | 43.21% | 6.79 pp | -38 | 23 | -1.65 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 280 | 120 | 160 | 42.86% | 42.92% | 42.86% | 7.14 pp | -40 | 23 | -1.74 |
| BTC Daily | nn | NN | 282 | 129 | 153 | 45.74% | 45.00% | 45.74% | 4.26 pp | -24 | 13 | -1.85 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 7 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 282 | 128 | 154 | 45.39% | 45.00% | 45.39% | 4.61 pp | -26 | 13 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 246 | 108 | 138 | 43.90% | 43.33% | 43.90% | 6.10 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 246 | 108 | 138 | 43.90% | 43.33% | 43.90% | 6.10 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 280 | 115 | 165 | 41.07% | 43.33% | 41.07% | 8.93 pp | -50 | 23 | -2.17 |
| BTC Hourly | transformer | Transformer | 256 | 116 | 140 | 45.31% | 45.83% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| Consolidated Hourly | xgb | XGBoost | 246 | 106 | 140 | 43.09% | 43.33% | 43.09% | 6.91 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 246 | 106 | 140 | 43.09% | 43.33% | 43.09% | 6.91 pp | -34 | 15 | -2.27 |
| BTC Market Hours | lstm | LSTM | 280 | 115 | 165 | 41.07% | 42.50% | 41.07% | 8.93 pp | -50 | 22 | -2.27 |
| Consolidated Hourly | nn | NN | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 15 | -2.53 |
| Consolidated Daily/Hourly Refresh | nn | NN | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 35 | 54 | 39.33% | 39.33% | 39.33% | 10.67 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 256 | 108 | 148 | 42.19% | 42.50% | 42.19% | 7.81 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 11 | -4.18 |
| BTC Daily | transformer | Transformer | 282 | 113 | 169 | 40.07% | 37.50% | 40.07% | 9.93 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 282 | 106 | 176 | 37.59% | 36.67% | 37.59% | 12.41 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 292 | 108 | 184 | 36.99% | 37.08% | 36.99% | 13.01 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 282 | 101 | 181 | 35.82% | 36.25% | 35.82% | 14.18 pp | -80 | 13 | -6.15 |
| BTC Hourly | lstm | LSTM | 256 | 94 | 162 | 36.72% | 36.25% | 36.72% | 13.28 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 256 | 89 | 167 | 34.77% | 35.00% | 34.77% | 15.23 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 256 | 123 | 133 | 48.05% | 47.92% | 48.05% | 1.95 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 256 | 116 | 140 | 45.31% | 45.83% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 256 | 108 | 148 | 42.19% | 42.50% | 42.19% | 7.81 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 11 | -4.18 |
| BTC Hourly | lstm | LSTM | 256 | 94 | 162 | 36.72% | 36.25% | 36.72% | 13.28 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 256 | 89 | 167 | 34.77% | 35.00% | 34.77% | 15.23 pp | -78 | 11 | -7.09 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 282 | 129 | 153 | 45.74% | 45.00% | 45.74% | 4.26 pp | -24 | 13 | -1.85 |
| BTC Daily | mlp_sklearn | MLPClassifier | 282 | 128 | 154 | 45.39% | 45.00% | 45.39% | 4.61 pp | -26 | 13 | -2.00 |
| BTC Daily | transformer | Transformer | 282 | 113 | 169 | 40.07% | 37.50% | 40.07% | 9.93 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 282 | 106 | 176 | 37.59% | 36.67% | 37.59% | 12.41 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 292 | 108 | 184 | 36.99% | 37.08% | 36.99% | 13.01 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 282 | 101 | 181 | 35.82% | 36.25% | 35.82% | 14.18 pp | -80 | 13 | -6.15 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 280 | 142 | 138 | 50.71% | 50.00% | 50.71% | 0.71 pp | 4 | 22 | 0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 280 | 129 | 151 | 46.07% | 46.25% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 280 | 129 | 151 | 46.07% | 45.42% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 280 | 126 | 154 | 45.00% | 45.00% | 45.00% | 5.00 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 280 | 124 | 156 | 44.29% | 42.92% | 44.29% | 5.71 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 280 | 115 | 165 | 41.07% | 42.50% | 41.07% | 8.93 pp | -50 | 22 | -2.27 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 280 | 138 | 142 | 49.29% | 48.75% | 49.29% | 0.71 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 280 | 135 | 145 | 48.21% | 48.33% | 48.21% | 1.79 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | nn | NN | 280 | 134 | 146 | 47.86% | 48.75% | 47.86% | 2.14 pp | -12 | 23 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 280 | 121 | 159 | 43.21% | 42.50% | 43.21% | 6.79 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | rf | RandomForest | 280 | 120 | 160 | 42.86% | 42.92% | 42.86% | 7.14 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 280 | 115 | 165 | 41.07% | 43.33% | 41.07% | 8.93 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 246 | 115 | 131 | 46.75% | 46.67% | 46.75% | 3.25 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 246 | 113 | 133 | 45.93% | 46.25% | 45.93% | 4.07 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 15 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 246 | 108 | 138 | 43.90% | 43.33% | 43.90% | 6.10 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 246 | 106 | 140 | 43.09% | 43.33% | 43.09% | 6.91 pp | -34 | 15 | -2.27 |
| Consolidated Hourly | nn | NN | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 15 | -2.53 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 246 | 115 | 131 | 46.75% | 46.67% | 46.75% | 3.25 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 246 | 113 | 133 | 45.93% | 46.25% | 45.93% | 4.07 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 246 | 112 | 134 | 45.53% | 45.00% | 45.53% | 4.47 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 246 | 108 | 138 | 43.90% | 43.33% | 43.90% | 6.10 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 246 | 106 | 140 | 43.09% | 43.33% | 43.09% | 6.91 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 246 | 104 | 142 | 42.28% | 42.92% | 42.28% | 7.72 pp | -38 | 15 | -2.53 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 35 | 54 | 39.33% | 39.33% | 39.33% | 10.67 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
