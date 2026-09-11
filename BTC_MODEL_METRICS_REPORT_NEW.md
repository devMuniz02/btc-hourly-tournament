# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T03:53:23.789294+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 353 | 293 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 528 | 281 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 528 | 281 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 247 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 21:00:00+00:00 | 247 | 89 | 158 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 281 | 143 | 138 | 50.89% | 50.00% | 50.89% | 0.89 pp | 5 | 22 | 0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 281 | 139 | 142 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 281 | 135 | 146 | 48.04% | 48.75% | 48.04% | 1.96 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | transformer | Transformer | 281 | 135 | 146 | 48.04% | 48.33% | 48.04% | 1.96 pp | -11 | 23 | -0.48 |
| Consolidated Hourly | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 247 | 118 | 129 | 47.77% | 47.92% | 47.77% | 2.23 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 257 | 124 | 133 | 48.25% | 48.33% | 48.25% | 1.75 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 281 | 130 | 151 | 46.26% | 46.25% | 46.26% | 3.74 pp | -21 | 22 | -0.95 |
| BTC Market Hours | transformer | Transformer | 281 | 130 | 151 | 46.26% | 45.83% | 46.26% | 3.74 pp | -21 | 22 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 247 | 114 | 133 | 46.15% | 45.83% | 46.15% | 3.85 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 7 | -1.29 |
| BTC Market Hours | xgb | XGBoost | 281 | 126 | 155 | 44.84% | 45.00% | 44.84% | 5.16 pp | -29 | 22 | -1.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 247 | 113 | 134 | 45.75% | 46.25% | 45.75% | 4.25 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 281 | 124 | 157 | 44.13% | 42.92% | 44.13% | 5.87 pp | -33 | 22 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 247 | 111 | 136 | 44.94% | 44.58% | 44.94% | 5.06 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 281 | 121 | 160 | 43.06% | 42.92% | 43.06% | 6.94 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 281 | 121 | 160 | 43.06% | 42.50% | 43.06% | 6.94 pp | -39 | 23 | -1.70 |
| BTC Daily | nn | NN | 283 | 129 | 154 | 45.58% | 45.00% | 45.58% | 4.42 pp | -25 | 13 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 283 | 128 | 155 | 45.23% | 45.00% | 45.23% | 4.77 pp | -27 | 13 | -2.08 |
| BTC Hourly | transformer | Transformer | 257 | 117 | 140 | 45.53% | 45.83% | 45.53% | 4.47 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 281 | 115 | 166 | 40.93% | 43.33% | 40.93% | 9.07 pp | -51 | 23 | -2.22 |
| BTC Market Hours | lstm | LSTM | 281 | 116 | 165 | 41.28% | 42.92% | 41.28% | 8.72 pp | -49 | 22 | -2.23 |
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
| BTC Daily | transformer | Transformer | 283 | 114 | 169 | 40.28% | 37.92% | 40.28% | 9.72 pp | -55 | 13 | -4.23 |
| BTC Hourly | rf | RandomForest | 257 | 105 | 152 | 40.86% | 40.83% | 40.86% | 9.14 pp | -47 | 11 | -4.27 |
| BTC Daily | rf | RandomForest | 283 | 107 | 176 | 37.81% | 37.08% | 37.81% | 12.19 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 293 | 109 | 184 | 37.20% | 37.50% | 37.20% | 12.80 pp | -75 | 14 | -5.36 |
| BTC Hourly | lstm | LSTM | 257 | 95 | 162 | 36.96% | 36.67% | 36.96% | 13.04 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 283 | 101 | 182 | 35.69% | 36.25% | 35.69% | 14.31 pp | -81 | 13 | -6.23 |
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
| BTC Daily | nn | NN | 283 | 129 | 154 | 45.58% | 45.00% | 45.58% | 4.42 pp | -25 | 13 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 283 | 128 | 155 | 45.23% | 45.00% | 45.23% | 4.77 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 283 | 114 | 169 | 40.28% | 37.92% | 40.28% | 9.72 pp | -55 | 13 | -4.23 |
| BTC Daily | rf | RandomForest | 283 | 107 | 176 | 37.81% | 37.08% | 37.81% | 12.19 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 293 | 109 | 184 | 37.20% | 37.50% | 37.20% | 12.80 pp | -75 | 14 | -5.36 |
| BTC Daily | lstm | LSTM | 283 | 101 | 182 | 35.69% | 36.25% | 35.69% | 14.31 pp | -81 | 13 | -6.23 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 281 | 143 | 138 | 50.89% | 50.00% | 50.89% | 0.89 pp | 5 | 22 | 0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 281 | 130 | 151 | 46.26% | 46.25% | 46.26% | 3.74 pp | -21 | 22 | -0.95 |
| BTC Market Hours | transformer | Transformer | 281 | 130 | 151 | 46.26% | 45.83% | 46.26% | 3.74 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 281 | 126 | 155 | 44.84% | 45.00% | 44.84% | 5.16 pp | -29 | 22 | -1.32 |
| BTC Market Hours | rf | RandomForest | 281 | 124 | 157 | 44.13% | 42.92% | 44.13% | 5.87 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 281 | 116 | 165 | 41.28% | 42.92% | 41.28% | 8.72 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 281 | 139 | 142 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 281 | 135 | 146 | 48.04% | 48.75% | 48.04% | 1.96 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | transformer | Transformer | 281 | 135 | 146 | 48.04% | 48.33% | 48.04% | 1.96 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | rf | RandomForest | 281 | 121 | 160 | 43.06% | 42.92% | 43.06% | 6.94 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 281 | 121 | 160 | 43.06% | 42.50% | 43.06% | 6.94 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 281 | 115 | 166 | 40.93% | 43.33% | 40.93% | 9.07 pp | -51 | 23 | -2.22 |

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
