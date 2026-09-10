# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T15:23:36.472136+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 345 | 285 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 510 | 273 | 237 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 14:00:00+00:00 | 510 | 273 | 237 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 273 | 141 | 132 | 51.65% | 51.25% | 51.65% | 1.65 pp | 9 | 21 | 0.43 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 273 | 133 | 140 | 48.72% | 47.92% | 48.72% | 1.28 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | transformer | Transformer | 273 | 132 | 141 | 48.35% | 47.92% | 48.35% | 1.65 pp | -9 | 22 | -0.41 |
| BTC Market Hours Daily | nn | NN | 273 | 130 | 143 | 47.62% | 47.92% | 47.62% | 2.38 pp | -13 | 22 | -0.59 |
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 249 | 120 | 129 | 48.19% | 47.92% | 48.19% | 1.81 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 273 | 127 | 146 | 46.52% | 46.25% | 46.52% | 3.48 pp | -19 | 21 | -0.90 |
| BTC Market Hours | transformer | Transformer | 273 | 127 | 146 | 46.52% | 46.25% | 46.52% | 3.48 pp | -19 | 21 | -0.90 |
| BTC Market Hours | xgb | XGBoost | 273 | 124 | 149 | 45.42% | 45.00% | 45.42% | 4.58 pp | -25 | 21 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 273 | 120 | 153 | 43.96% | 42.08% | 43.96% | 6.04 pp | -33 | 21 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 273 | 119 | 154 | 43.59% | 42.50% | 43.59% | 6.41 pp | -35 | 22 | -1.59 |
| BTC Daily | nn | NN | 275 | 127 | 148 | 46.18% | 45.42% | 46.18% | 3.82 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 273 | 116 | 157 | 42.49% | 41.67% | 42.49% | 7.51 pp | -41 | 22 | -1.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 275 | 125 | 150 | 45.45% | 45.00% | 45.45% | 4.55 pp | -25 | 12 | -2.08 |
| BTC Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 46.25% | 45.38% | 4.62 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 273 | 113 | 160 | 41.39% | 42.92% | 41.39% | 8.61 pp | -47 | 22 | -2.14 |
| BTC Market Hours | lstm | LSTM | 273 | 113 | 160 | 41.39% | 42.50% | 41.39% | 8.61 pp | -47 | 21 | -2.24 |
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
| BTC Daily | transformer | Transformer | 275 | 111 | 164 | 40.36% | 37.92% | 40.36% | 9.64 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 275 | 104 | 171 | 37.82% | 37.08% | 37.82% | 12.18 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 285 | 105 | 180 | 36.84% | 36.67% | 36.84% | 13.16 pp | -75 | 13 | -5.77 |
| BTC Hourly | lstm | LSTM | 249 | 92 | 157 | 36.95% | 36.67% | 36.95% | 13.05 pp | -65 | 11 | -5.91 |
| BTC Hourly | xgb | XGBoost | 249 | 88 | 161 | 35.34% | 36.25% | 35.34% | 14.66 pp | -73 | 11 | -6.64 |
| BTC Daily | lstm | LSTM | 275 | 96 | 179 | 34.91% | 35.00% | 34.91% | 15.09 pp | -83 | 12 | -6.92 |

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
| BTC Daily | nn | NN | 275 | 127 | 148 | 46.18% | 45.42% | 46.18% | 3.82 pp | -21 | 12 | -1.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 275 | 125 | 150 | 45.45% | 45.00% | 45.45% | 4.55 pp | -25 | 12 | -2.08 |
| BTC Daily | transformer | Transformer | 275 | 111 | 164 | 40.36% | 37.92% | 40.36% | 9.64 pp | -53 | 12 | -4.42 |
| BTC Daily | rf | RandomForest | 275 | 104 | 171 | 37.82% | 37.08% | 37.82% | 12.18 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 285 | 105 | 180 | 36.84% | 36.67% | 36.84% | 13.16 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 275 | 96 | 179 | 34.91% | 35.00% | 34.91% | 15.09 pp | -83 | 12 | -6.92 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 273 | 141 | 132 | 51.65% | 51.25% | 51.65% | 1.65 pp | 9 | 21 | 0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 273 | 127 | 146 | 46.52% | 46.25% | 46.52% | 3.48 pp | -19 | 21 | -0.90 |
| BTC Market Hours | transformer | Transformer | 273 | 127 | 146 | 46.52% | 46.25% | 46.52% | 3.48 pp | -19 | 21 | -0.90 |
| BTC Market Hours | xgb | XGBoost | 273 | 124 | 149 | 45.42% | 45.00% | 45.42% | 4.58 pp | -25 | 21 | -1.19 |
| BTC Market Hours | rf | RandomForest | 273 | 120 | 153 | 43.96% | 42.08% | 43.96% | 6.04 pp | -33 | 21 | -1.57 |
| BTC Market Hours | lstm | LSTM | 273 | 113 | 160 | 41.39% | 42.50% | 41.39% | 8.61 pp | -47 | 21 | -2.24 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 273 | 133 | 140 | 48.72% | 47.92% | 48.72% | 1.28 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | transformer | Transformer | 273 | 132 | 141 | 48.35% | 47.92% | 48.35% | 1.65 pp | -9 | 22 | -0.41 |
| BTC Market Hours Daily | nn | NN | 273 | 130 | 143 | 47.62% | 47.92% | 47.62% | 2.38 pp | -13 | 22 | -0.59 |
| BTC Market Hours Daily | xgb | XGBoost | 273 | 119 | 154 | 43.59% | 42.50% | 43.59% | 6.41 pp | -35 | 22 | -1.59 |
| BTC Market Hours Daily | rf | RandomForest | 273 | 116 | 157 | 42.49% | 41.67% | 42.49% | 7.51 pp | -41 | 22 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 273 | 113 | 160 | 41.39% | 42.92% | 41.39% | 8.61 pp | -47 | 22 | -2.14 |

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
