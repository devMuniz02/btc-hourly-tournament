# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T14:10:24.719735+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 324 | 264 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 360 | 300 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 537 | 288 | 249 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 13:00:00+00:00 | 536 | 287 | 249 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 251 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 00:00:00+00:00 | 251 | 92 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 287 | 141 | 146 | 49.13% | 48.75% | 49.13% | 0.87 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | nn | NN | 287 | 139 | 148 | 48.43% | 50.00% | 48.43% | 1.57 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 287 | 138 | 149 | 48.08% | 48.33% | 48.08% | 1.92 pp | -11 | 23 | -0.48 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 264 | 127 | 137 | 48.11% | 47.50% | 48.11% | 1.89 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| BTC Daily | nn | NN | 290 | 135 | 155 | 46.55% | 46.25% | 46.55% | 3.45 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| BTC Hourly | transformer | Transformer | 264 | 123 | 141 | 46.59% | 47.08% | 46.59% | 3.41 pp | -18 | 11 | -1.64 |
| BTC Market Hours Daily | xgb | XGBoost | 287 | 124 | 163 | 43.21% | 43.75% | 43.21% | 6.79 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | rf | RandomForest | 287 | 123 | 164 | 42.86% | 42.92% | 42.86% | 7.14 pp | -41 | 23 | -1.78 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 287 | 118 | 169 | 41.11% | 43.75% | 41.11% | 8.89 pp | -51 | 23 | -2.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 43.75% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| BTC Hourly | nn | NN | 264 | 110 | 154 | 41.67% | 40.83% | 41.67% | 8.33 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 290 | 117 | 173 | 40.34% | 37.92% | 40.34% | 9.66 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 264 | 106 | 158 | 40.15% | 40.42% | 40.15% | 9.85 pp | -52 | 11 | -4.73 |
| BTC Daily | rf | RandomForest | 290 | 111 | 179 | 38.28% | 37.08% | 38.28% | 11.72 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 300 | 113 | 187 | 37.67% | 37.92% | 37.67% | 12.33 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 290 | 103 | 187 | 35.52% | 35.83% | 35.52% | 14.48 pp | -84 | 13 | -6.46 |
| BTC Hourly | lstm | LSTM | 264 | 95 | 169 | 35.98% | 34.58% | 35.98% | 14.02 pp | -74 | 11 | -6.73 |
| BTC Hourly | xgb | XGBoost | 264 | 91 | 173 | 34.47% | 34.58% | 34.47% | 15.53 pp | -82 | 11 | -7.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 264 | 127 | 137 | 48.11% | 47.50% | 48.11% | 1.89 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 264 | 123 | 141 | 46.59% | 47.08% | 46.59% | 3.41 pp | -18 | 11 | -1.64 |
| BTC Hourly | nn | NN | 264 | 110 | 154 | 41.67% | 40.83% | 41.67% | 8.33 pp | -44 | 11 | -4.00 |
| BTC Hourly | rf | RandomForest | 264 | 106 | 158 | 40.15% | 40.42% | 40.15% | 9.85 pp | -52 | 11 | -4.73 |
| BTC Hourly | lstm | LSTM | 264 | 95 | 169 | 35.98% | 34.58% | 35.98% | 14.02 pp | -74 | 11 | -6.73 |
| BTC Hourly | xgb | XGBoost | 264 | 91 | 173 | 34.47% | 34.58% | 34.47% | 15.53 pp | -82 | 11 | -7.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 290 | 135 | 155 | 46.55% | 46.25% | 46.55% | 3.45 pp | -20 | 13 | -1.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 43.75% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
| BTC Daily | transformer | Transformer | 290 | 117 | 173 | 40.34% | 37.92% | 40.34% | 9.66 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 290 | 111 | 179 | 38.28% | 37.08% | 38.28% | 11.72 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 300 | 113 | 187 | 37.67% | 37.92% | 37.67% | 12.33 pp | -74 | 14 | -5.29 |
| BTC Daily | lstm | LSTM | 290 | 103 | 187 | 35.52% | 35.83% | 35.52% | 14.48 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 287 | 141 | 146 | 49.13% | 48.75% | 49.13% | 0.87 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | nn | NN | 287 | 139 | 148 | 48.43% | 50.00% | 48.43% | 1.57 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 287 | 138 | 149 | 48.08% | 48.33% | 48.08% | 1.92 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | xgb | XGBoost | 287 | 124 | 163 | 43.21% | 43.75% | 43.21% | 6.79 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | rf | RandomForest | 287 | 123 | 164 | 42.86% | 42.92% | 42.86% | 7.14 pp | -41 | 23 | -1.78 |
| BTC Market Hours Daily | lstm | LSTM | 287 | 118 | 169 | 41.11% | 43.75% | 41.11% | 8.89 pp | -51 | 23 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 251 | 118 | 133 | 47.01% | 46.67% | 47.01% | 2.99 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 251 | 115 | 136 | 45.82% | 44.58% | 45.82% | 4.18 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 251 | 113 | 138 | 45.02% | 45.00% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 251 | 113 | 138 | 45.02% | 44.17% | 45.02% | 4.98 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 251 | 103 | 148 | 41.04% | 40.83% | 41.04% | 8.96 pp | -45 | 16 | -2.81 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 92 | 34 | 58 | 36.96% | 36.96% | 36.96% | 13.04 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 8 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
