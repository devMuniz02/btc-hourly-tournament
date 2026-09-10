# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T14:37:33.540901+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 308 | 248 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 344 | 284 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 13:00:00+00:00 | 508 | 272 | 236 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 13:00:00+00:00 | 508 | 272 | 236 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 239 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T17:00:00+00:00 | 240 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 272 | 141 | 131 | 51.84% | 51.67% | 51.84% | 1.84 pp | 10 | 21 | 0.48 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 272 | 133 | 139 | 48.90% | 47.92% | 48.90% | 1.10 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 272 | 132 | 140 | 48.53% | 47.92% | 48.53% | 1.47 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 272 | 130 | 142 | 47.79% | 48.33% | 47.79% | 2.21 pp | -12 | 22 | -0.55 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 248 | 120 | 128 | 48.39% | 48.33% | 48.39% | 1.61 pp | -8 | 11 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 272 | 126 | 146 | 46.32% | 46.25% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | transformer | Transformer | 272 | 126 | 146 | 46.32% | 45.83% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 21 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 272 | 119 | 153 | 43.75% | 42.50% | 43.75% | 6.25 pp | -34 | 22 | -1.55 |
| BTC Market Hours | rf | RandomForest | 272 | 119 | 153 | 43.75% | 42.08% | 43.75% | 6.25 pp | -34 | 21 | -1.62 |
| BTC Daily | nn | NN | 274 | 127 | 147 | 46.35% | 45.42% | 46.35% | 3.65 pp | -20 | 12 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 272 | 116 | 156 | 42.65% | 41.67% | 42.65% | 7.35 pp | -40 | 22 | -1.82 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 125 | 149 | 45.62% | 45.00% | 45.62% | 4.38 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| BTC Hourly | transformer | Transformer | 248 | 112 | 136 | 45.16% | 46.25% | 45.16% | 4.84 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 272 | 112 | 160 | 41.18% | 42.50% | 41.18% | 8.82 pp | -48 | 22 | -2.18 |
| BTC Market Hours | lstm | LSTM | 272 | 113 | 159 | 41.54% | 42.50% | 41.54% | 8.46 pp | -46 | 21 | -2.19 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 34 | 52 | 39.53% | 39.53% | 39.53% | 10.47 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 248 | 105 | 143 | 42.34% | 42.50% | 42.34% | 7.66 pp | -38 | 11 | -3.45 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 86 | 30 | 56 | 34.88% | 34.88% | 34.88% | 15.12 pp | -26 | 7 | -3.71 |
| BTC Hourly | rf | RandomForest | 248 | 103 | 145 | 41.53% | 42.50% | 41.53% | 8.47 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 274 | 111 | 163 | 40.51% | 37.92% | 40.51% | 9.49 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 274 | 104 | 170 | 37.96% | 37.50% | 37.96% | 12.04 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 284 | 105 | 179 | 36.97% | 36.67% | 36.97% | 13.03 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 248 | 92 | 156 | 37.10% | 36.67% | 37.10% | 12.90 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 248 | 88 | 160 | 35.48% | 36.25% | 35.48% | 14.52 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 274 | 96 | 178 | 35.04% | 35.00% | 35.04% | 14.96 pp | -82 | 12 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 248 | 120 | 128 | 48.39% | 48.33% | 48.39% | 1.61 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 248 | 112 | 136 | 45.16% | 46.25% | 45.16% | 4.84 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 248 | 105 | 143 | 42.34% | 42.50% | 42.34% | 7.66 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 248 | 103 | 145 | 41.53% | 42.50% | 41.53% | 8.47 pp | -42 | 11 | -3.82 |
| BTC Hourly | lstm | LSTM | 248 | 92 | 156 | 37.10% | 36.67% | 37.10% | 12.90 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 248 | 88 | 160 | 35.48% | 36.25% | 35.48% | 14.52 pp | -72 | 11 | -6.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 274 | 127 | 147 | 46.35% | 45.42% | 46.35% | 3.65 pp | -20 | 12 | -1.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 125 | 149 | 45.62% | 45.00% | 45.62% | 4.38 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 274 | 111 | 163 | 40.51% | 37.92% | 40.51% | 9.49 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 274 | 104 | 170 | 37.96% | 37.50% | 37.96% | 12.04 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 284 | 105 | 179 | 36.97% | 36.67% | 36.97% | 13.03 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 274 | 96 | 178 | 35.04% | 35.00% | 35.04% | 14.96 pp | -82 | 12 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 272 | 141 | 131 | 51.84% | 51.67% | 51.84% | 1.84 pp | 10 | 21 | 0.48 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 272 | 126 | 146 | 46.32% | 46.25% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | transformer | Transformer | 272 | 126 | 146 | 46.32% | 45.83% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 21 | -1.14 |
| BTC Market Hours | rf | RandomForest | 272 | 119 | 153 | 43.75% | 42.08% | 43.75% | 6.25 pp | -34 | 21 | -1.62 |
| BTC Market Hours | lstm | LSTM | 272 | 113 | 159 | 41.54% | 42.50% | 41.54% | 8.46 pp | -46 | 21 | -2.19 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 272 | 133 | 139 | 48.90% | 47.92% | 48.90% | 1.10 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 272 | 132 | 140 | 48.53% | 47.92% | 48.53% | 1.47 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 272 | 130 | 142 | 47.79% | 48.33% | 47.79% | 2.21 pp | -12 | 22 | -0.55 |
| BTC Market Hours Daily | xgb | XGBoost | 272 | 119 | 153 | 43.75% | 42.50% | 43.75% | 6.25 pp | -34 | 22 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 272 | 116 | 156 | 42.65% | 41.67% | 42.65% | 7.35 pp | -40 | 22 | -1.82 |
| BTC Market Hours Daily | lstm | LSTM | 272 | 112 | 160 | 41.18% | 42.50% | 41.18% | 8.82 pp | -48 | 22 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 112 | 127 | 46.86% | 46.86% | 46.86% | 3.14 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 108 | 131 | 45.19% | 45.19% | 45.19% | 4.81 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 104 | 135 | 43.51% | 43.51% | 43.51% | 6.49 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 39 | 47 | 45.35% | 45.35% | 45.35% | 4.65 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 36 | 50 | 41.86% | 41.86% | 41.86% | 8.14 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 34 | 52 | 39.53% | 39.53% | 39.53% | 10.47 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 86 | 30 | 56 | 34.88% | 34.88% | 34.88% | 15.12 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
