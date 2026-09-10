# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T15:04:25.825921+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 309 | 249 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 344 | 284 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 509 | 272 | 237 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 509 | 272 | 237 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 272 | 141 | 131 | 51.84% | 51.67% | 51.84% | 1.84 pp | 10 | 21 | 0.48 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 272 | 133 | 139 | 48.90% | 47.92% | 48.90% | 1.10 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 272 | 132 | 140 | 48.53% | 47.92% | 48.53% | 1.47 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 272 | 130 | 142 | 47.79% | 48.33% | 47.79% | 2.21 pp | -12 | 22 | -0.55 |
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 249 | 120 | 129 | 48.19% | 47.92% | 48.19% | 1.81 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 272 | 126 | 146 | 46.32% | 46.25% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | transformer | Transformer | 272 | 126 | 146 | 46.32% | 45.83% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 21 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | xgb | XGBoost | 272 | 119 | 153 | 43.75% | 42.50% | 43.75% | 6.25 pp | -34 | 22 | -1.55 |
| BTC Market Hours | rf | RandomForest | 272 | 119 | 153 | 43.75% | 42.08% | 43.75% | 6.25 pp | -34 | 21 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 272 | 116 | 156 | 42.65% | 41.67% | 42.65% | 7.35 pp | -40 | 22 | -1.82 |
| BTC Daily | nn | NN | 274 | 126 | 148 | 45.99% | 45.42% | 45.99% | 4.01 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 46.25% | 45.38% | 4.62 pp | -23 | 11 | -2.09 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 124 | 150 | 45.26% | 45.00% | 45.26% | 4.74 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 272 | 112 | 160 | 41.18% | 42.50% | 41.18% | 8.82 pp | -48 | 22 | -2.18 |
| BTC Market Hours | lstm | LSTM | 272 | 113 | 159 | 41.54% | 42.50% | 41.54% | 8.46 pp | -46 | 21 | -2.19 |
| Consolidated Market Hours | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| BTC Hourly | nn | NN | 249 | 105 | 144 | 42.17% | 42.50% | 42.17% | 7.83 pp | -39 | 11 | -3.55 |
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 249 | 103 | 146 | 41.37% | 42.08% | 41.37% | 8.63 pp | -43 | 11 | -3.91 |
| BTC Daily | transformer | Transformer | 274 | 110 | 164 | 40.15% | 37.92% | 40.15% | 9.85 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 274 | 103 | 171 | 37.59% | 37.08% | 37.59% | 12.41 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 284 | 104 | 180 | 36.62% | 36.67% | 36.62% | 13.38 pp | -76 | 13 | -5.85 |
| BTC Hourly | lstm | LSTM | 249 | 92 | 157 | 36.95% | 36.67% | 36.95% | 13.05 pp | -65 | 11 | -5.91 |
| BTC Hourly | xgb | XGBoost | 249 | 88 | 161 | 35.34% | 36.25% | 35.34% | 14.66 pp | -73 | 11 | -6.64 |
| BTC Daily | lstm | LSTM | 274 | 96 | 178 | 35.04% | 35.00% | 35.04% | 14.96 pp | -82 | 12 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 249 | 120 | 129 | 48.19% | 47.92% | 48.19% | 1.81 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 46.25% | 45.38% | 4.62 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 249 | 105 | 144 | 42.17% | 42.50% | 42.17% | 7.83 pp | -39 | 11 | -3.55 |
| BTC Hourly | rf | RandomForest | 249 | 103 | 146 | 41.37% | 42.08% | 41.37% | 8.63 pp | -43 | 11 | -3.91 |
| BTC Hourly | lstm | LSTM | 249 | 92 | 157 | 36.95% | 36.67% | 36.95% | 13.05 pp | -65 | 11 | -5.91 |
| BTC Hourly | xgb | XGBoost | 249 | 88 | 161 | 35.34% | 36.25% | 35.34% | 14.66 pp | -73 | 11 | -6.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 274 | 126 | 148 | 45.99% | 45.42% | 45.99% | 4.01 pp | -22 | 12 | -1.83 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 124 | 150 | 45.26% | 45.00% | 45.26% | 4.74 pp | -26 | 12 | -2.17 |
| BTC Daily | transformer | Transformer | 274 | 110 | 164 | 40.15% | 37.92% | 40.15% | 9.85 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 274 | 103 | 171 | 37.59% | 37.08% | 37.59% | 12.41 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 284 | 104 | 180 | 36.62% | 36.67% | 36.62% | 13.38 pp | -76 | 13 | -5.85 |
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
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 239 | 100 | 139 | 41.84% | 41.84% | 41.84% | 8.16 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 239 | 96 | 143 | 40.17% | 40.17% | 40.17% | 9.83 pp | -47 | 15 | -3.13 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 85 | 34 | 51 | 40.00% | 40.00% | 40.00% | 10.00 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 85 | 32 | 53 | 37.65% | 37.65% | 37.65% | 12.35 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 85 | 31 | 54 | 36.47% | 36.47% | 36.47% | 13.53 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
