# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T01:16:07.820486+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 315 | 255 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 351 | 291 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 526 | 279 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 526 | 279 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 245 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T20:00:00+00:00 | 246 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 279 | 142 | 137 | 50.90% | 50.42% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 279 | 137 | 142 | 49.10% | 48.33% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 279 | 134 | 145 | 48.03% | 47.92% | 48.03% | 1.97 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | nn | NN | 279 | 133 | 146 | 47.67% | 48.33% | 47.67% | 2.33 pp | -13 | 23 | -0.57 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 255 | 123 | 132 | 48.24% | 48.33% | 48.24% | 1.76 pp | -9 | 11 | -0.82 |
| BTC Market Hours | transformer | Transformer | 279 | 129 | 150 | 46.24% | 45.83% | 46.24% | 3.76 pp | -21 | 22 | -0.95 |
| Consolidated Hourly | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 279 | 128 | 151 | 45.88% | 45.83% | 45.88% | 4.12 pp | -23 | 22 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 279 | 126 | 153 | 45.16% | 45.00% | 45.16% | 4.84 pp | -27 | 22 | -1.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 245 | 112 | 133 | 45.71% | 45.42% | 45.71% | 4.29 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 112 | 133 | 45.71% | 45.42% | 45.71% | 4.29 pp | -21 | 15 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 279 | 123 | 156 | 44.09% | 42.92% | 44.09% | 5.91 pp | -33 | 22 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 279 | 120 | 159 | 43.01% | 42.50% | 43.01% | 6.99 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Daily | nn | NN | 281 | 129 | 152 | 45.91% | 45.00% | 45.91% | 4.09 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 279 | 119 | 160 | 42.65% | 42.50% | 42.65% | 7.35 pp | -41 | 23 | -1.78 |
| Consolidated Market Hours Daily | rf | RandomForest | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 7 | -1.86 |
| Consolidated Hourly | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| BTC Daily | mlp_sklearn | MLPClassifier | 281 | 127 | 154 | 45.20% | 44.58% | 45.20% | 4.80 pp | -27 | 13 | -2.08 |
| Consolidated Hourly | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 279 | 114 | 165 | 40.86% | 42.92% | 40.86% | 9.14 pp | -51 | 23 | -2.22 |
| BTC Market Hours | lstm | LSTM | 279 | 115 | 164 | 41.22% | 42.50% | 41.22% | 8.78 pp | -49 | 22 | -2.23 |
| BTC Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 45.83% | 45.10% | 4.90 pp | -25 | 11 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 89 | 35 | 54 | 39.33% | 39.33% | 39.33% | 10.67 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 255 | 107 | 148 | 41.96% | 42.50% | 41.96% | 8.04 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 281 | 113 | 168 | 40.21% | 37.50% | 40.21% | 9.79 pp | -55 | 13 | -4.23 |
| BTC Daily | rf | RandomForest | 281 | 106 | 175 | 37.72% | 36.67% | 37.72% | 12.28 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 291 | 108 | 183 | 37.11% | 37.08% | 37.11% | 12.89 pp | -75 | 14 | -5.36 |
| BTC Hourly | lstm | LSTM | 255 | 94 | 161 | 36.86% | 36.25% | 36.86% | 13.14 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 281 | 100 | 181 | 35.59% | 35.83% | 35.59% | 14.41 pp | -81 | 13 | -6.23 |
| BTC Hourly | xgb | XGBoost | 255 | 89 | 166 | 34.90% | 35.42% | 34.90% | 15.10 pp | -77 | 11 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 255 | 123 | 132 | 48.24% | 48.33% | 48.24% | 1.76 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 45.83% | 45.10% | 4.90 pp | -25 | 11 | -2.27 |
| BTC Hourly | nn | NN | 255 | 107 | 148 | 41.96% | 42.50% | 41.96% | 8.04 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 11 | -4.09 |
| BTC Hourly | lstm | LSTM | 255 | 94 | 161 | 36.86% | 36.25% | 36.86% | 13.14 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 255 | 89 | 166 | 34.90% | 35.42% | 34.90% | 15.10 pp | -77 | 11 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 281 | 129 | 152 | 45.91% | 45.00% | 45.91% | 4.09 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 281 | 127 | 154 | 45.20% | 44.58% | 45.20% | 4.80 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 281 | 113 | 168 | 40.21% | 37.50% | 40.21% | 9.79 pp | -55 | 13 | -4.23 |
| BTC Daily | rf | RandomForest | 281 | 106 | 175 | 37.72% | 36.67% | 37.72% | 12.28 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 291 | 108 | 183 | 37.11% | 37.08% | 37.11% | 12.89 pp | -75 | 14 | -5.36 |
| BTC Daily | lstm | LSTM | 281 | 100 | 181 | 35.59% | 35.83% | 35.59% | 14.41 pp | -81 | 13 | -6.23 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 279 | 142 | 137 | 50.90% | 50.42% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours | transformer | Transformer | 279 | 129 | 150 | 46.24% | 45.83% | 46.24% | 3.76 pp | -21 | 22 | -0.95 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 279 | 128 | 151 | 45.88% | 45.83% | 45.88% | 4.12 pp | -23 | 22 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 279 | 126 | 153 | 45.16% | 45.00% | 45.16% | 4.84 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 279 | 123 | 156 | 44.09% | 42.92% | 44.09% | 5.91 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 279 | 115 | 164 | 41.22% | 42.50% | 41.22% | 8.78 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 279 | 137 | 142 | 49.10% | 48.33% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 279 | 134 | 145 | 48.03% | 47.92% | 48.03% | 1.97 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | nn | NN | 279 | 133 | 146 | 47.67% | 48.33% | 47.67% | 2.33 pp | -13 | 23 | -0.57 |
| BTC Market Hours Daily | xgb | XGBoost | 279 | 120 | 159 | 43.01% | 42.50% | 43.01% | 6.99 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | rf | RandomForest | 279 | 119 | 160 | 42.65% | 42.50% | 42.65% | 7.35 pp | -41 | 23 | -1.78 |
| BTC Market Hours Daily | lstm | LSTM | 279 | 114 | 165 | 40.86% | 42.92% | 40.86% | 9.14 pp | -51 | 23 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 245 | 112 | 133 | 45.71% | 45.42% | 45.71% | 4.29 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Hourly | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 115 | 130 | 46.94% | 47.08% | 46.94% | 3.06 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 112 | 133 | 45.71% | 45.42% | 45.71% | 4.29 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 106 | 139 | 43.27% | 43.33% | 43.27% | 6.73 pp | -33 | 15 | -2.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 103 | 142 | 42.04% | 42.50% | 42.04% | 7.96 pp | -39 | 15 | -2.60 |

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
