# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T03:33:00.310178+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 317 | 257 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 352 | 292 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 527 | 280 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 527 | 280 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 280 | 142 | 138 | 50.71% | 50.00% | 50.71% | 0.71 pp | 4 | 22 | 0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 280 | 138 | 142 | 49.29% | 48.75% | 49.29% | 0.71 pp | -4 | 23 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 280 | 135 | 145 | 48.21% | 48.33% | 48.21% | 1.79 pp | -10 | 23 | -0.43 |
| BTC Market Hours Daily | nn | NN | 280 | 134 | 146 | 47.86% | 48.75% | 47.86% | 2.14 pp | -12 | 23 | -0.52 |
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 257 | 124 | 133 | 48.25% | 48.33% | 48.25% | 1.75 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 280 | 129 | 151 | 46.07% | 46.25% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| BTC Market Hours | transformer | Transformer | 280 | 129 | 151 | 46.07% | 45.42% | 46.07% | 3.93 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| BTC Market Hours | xgb | XGBoost | 280 | 126 | 154 | 45.00% | 45.00% | 45.00% | 5.00 pp | -28 | 22 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 280 | 124 | 156 | 44.29% | 42.92% | 44.29% | 5.71 pp | -32 | 22 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 280 | 121 | 159 | 43.21% | 42.50% | 43.21% | 6.79 pp | -38 | 23 | -1.65 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 280 | 120 | 160 | 42.86% | 42.92% | 42.86% | 7.14 pp | -40 | 23 | -1.74 |
| BTC Daily | nn | NN | 282 | 128 | 154 | 45.39% | 45.00% | 45.39% | 4.61 pp | -26 | 13 | -2.00 |
| BTC Hourly | transformer | Transformer | 257 | 117 | 140 | 45.53% | 45.83% | 45.53% | 4.47 pp | -23 | 11 | -2.09 |
| BTC Daily | mlp_sklearn | MLPClassifier | 282 | 127 | 155 | 45.04% | 45.00% | 45.04% | 4.96 pp | -28 | 13 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 280 | 115 | 165 | 41.07% | 43.33% | 41.07% | 8.93 pp | -50 | 23 | -2.17 |
| BTC Market Hours | lstm | LSTM | 280 | 115 | 165 | 41.07% | 42.50% | 41.07% | 8.93 pp | -50 | 22 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 257 | 109 | 148 | 42.41% | 42.92% | 42.41% | 7.59 pp | -39 | 11 | -3.55 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 257 | 105 | 152 | 40.86% | 40.83% | 40.86% | 9.14 pp | -47 | 11 | -4.27 |
| BTC Daily | transformer | Transformer | 282 | 113 | 169 | 40.07% | 37.92% | 40.07% | 9.93 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 282 | 106 | 176 | 37.59% | 37.08% | 37.59% | 12.41 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 292 | 108 | 184 | 36.99% | 37.50% | 36.99% | 13.01 pp | -76 | 14 | -5.43 |
| BTC Hourly | lstm | LSTM | 257 | 95 | 162 | 36.96% | 36.67% | 36.96% | 13.04 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 282 | 101 | 181 | 35.82% | 36.25% | 35.82% | 14.18 pp | -80 | 13 | -6.15 |
| BTC Hourly | xgb | XGBoost | 257 | 89 | 168 | 34.63% | 34.58% | 34.63% | 15.37 pp | -79 | 11 | -7.18 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 257 | 124 | 133 | 48.25% | 48.33% | 48.25% | 1.75 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 257 | 117 | 140 | 45.53% | 45.83% | 45.53% | 4.47 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 257 | 109 | 148 | 42.41% | 42.92% | 42.41% | 7.59 pp | -39 | 11 | -3.55 |
| BTC Hourly | rf | RandomForest | 257 | 105 | 152 | 40.86% | 40.83% | 40.86% | 9.14 pp | -47 | 11 | -4.27 |
| BTC Hourly | lstm | LSTM | 257 | 95 | 162 | 36.96% | 36.67% | 36.96% | 13.04 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 257 | 89 | 168 | 34.63% | 34.58% | 34.63% | 15.37 pp | -79 | 11 | -7.18 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 282 | 128 | 154 | 45.39% | 45.00% | 45.39% | 4.61 pp | -26 | 13 | -2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 282 | 127 | 155 | 45.04% | 45.00% | 45.04% | 4.96 pp | -28 | 13 | -2.15 |
| BTC Daily | transformer | Transformer | 282 | 113 | 169 | 40.07% | 37.92% | 40.07% | 9.93 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 282 | 106 | 176 | 37.59% | 37.08% | 37.59% | 12.41 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 292 | 108 | 184 | 36.99% | 37.50% | 36.99% | 13.01 pp | -76 | 14 | -5.43 |
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
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 247 | 103 | 144 | 41.70% | 41.67% | 41.70% | 8.30 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 247 | 100 | 147 | 40.49% | 41.25% | 40.49% | 9.51 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
