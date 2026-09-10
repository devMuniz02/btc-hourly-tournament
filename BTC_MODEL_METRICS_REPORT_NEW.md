# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T16:44:00.276362+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 310 | 250 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 346 | 286 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 15:00:00+00:00 | 512 | 274 | 238 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 15:00:00+00:00 | 511 | 273 | 238 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 239 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 17:00:00+00:00 | 239 | 85 | 154 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 274 | 141 | 133 | 51.46% | 51.25% | 51.46% | 1.46 pp | 8 | 22 | 0.36 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 273 | 133 | 140 | 48.72% | 47.92% | 48.72% | 1.28 pp | -7 | 22 | -0.32 |
| BTC Market Hours Daily | transformer | Transformer | 273 | 132 | 141 | 48.35% | 47.92% | 48.35% | 1.65 pp | -9 | 22 | -0.41 |
| BTC Market Hours Daily | nn | NN | 273 | 130 | 143 | 47.62% | 47.92% | 47.62% | 2.38 pp | -13 | 22 | -0.59 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 250 | 121 | 129 | 48.40% | 48.33% | 48.40% | 1.60 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 239 | 114 | 125 | 47.70% | 47.70% | 47.70% | 2.30 pp | -11 | 15 | -0.73 |
| BTC Market Hours | transformer | Transformer | 274 | 128 | 146 | 46.72% | 46.25% | 46.72% | 3.28 pp | -18 | 22 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 274 | 127 | 147 | 46.35% | 46.25% | 46.35% | 3.65 pp | -20 | 22 | -0.91 |
| BTC Market Hours | xgb | XGBoost | 274 | 124 | 150 | 45.26% | 45.00% | 45.26% | 4.74 pp | -26 | 22 | -1.18 |
| Consolidated Hourly | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 239 | 110 | 129 | 46.03% | 46.03% | 46.03% | 3.97 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 274 | 121 | 153 | 44.16% | 42.50% | 44.16% | 5.84 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | xgb | XGBoost | 273 | 119 | 154 | 43.59% | 42.50% | 43.59% | 6.41 pp | -35 | 22 | -1.59 |
| Consolidated Hourly | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 239 | 106 | 133 | 44.35% | 44.35% | 44.35% | 5.65 pp | -27 | 15 | -1.80 |
| BTC Daily | nn | NN | 276 | 127 | 149 | 46.01% | 45.42% | 46.01% | 3.99 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 273 | 116 | 157 | 42.49% | 41.67% | 42.49% | 7.51 pp | -41 | 22 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 273 | 113 | 160 | 41.39% | 42.92% | 41.39% | 8.61 pp | -47 | 22 | -2.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 12 | -2.17 |
| BTC Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 46.25% | 45.20% | 4.80 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 274 | 113 | 161 | 41.24% | 42.08% | 41.24% | 8.76 pp | -48 | 22 | -2.18 |
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
| Consolidated Market Hours | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 85 | 30 | 55 | 35.29% | 35.29% | 35.29% | 14.71 pp | -25 | 7 | -3.57 |
| BTC Hourly | nn | NN | 250 | 105 | 145 | 42.00% | 42.50% | 42.00% | 8.00 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 250 | 103 | 147 | 41.20% | 42.08% | 41.20% | 8.80 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 276 | 112 | 164 | 40.58% | 38.33% | 40.58% | 9.42 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 276 | 104 | 172 | 37.68% | 37.08% | 37.68% | 12.32 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 286 | 106 | 180 | 37.06% | 37.08% | 37.06% | 12.94 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 250 | 92 | 158 | 36.80% | 36.67% | 36.80% | 13.20 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 250 | 88 | 162 | 35.20% | 36.25% | 35.20% | 14.80 pp | -74 | 11 | -6.73 |
| BTC Daily | lstm | LSTM | 276 | 96 | 180 | 34.78% | 35.00% | 34.78% | 15.22 pp | -84 | 12 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 250 | 121 | 129 | 48.40% | 48.33% | 48.40% | 1.60 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 250 | 113 | 137 | 45.20% | 46.25% | 45.20% | 4.80 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 250 | 105 | 145 | 42.00% | 42.50% | 42.00% | 8.00 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 250 | 103 | 147 | 41.20% | 42.08% | 41.20% | 8.80 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 250 | 92 | 158 | 36.80% | 36.67% | 36.80% | 13.20 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 250 | 88 | 162 | 35.20% | 36.25% | 35.20% | 14.80 pp | -74 | 11 | -6.73 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 276 | 127 | 149 | 46.01% | 45.42% | 46.01% | 3.99 pp | -22 | 12 | -1.83 |
| BTC Daily | mlp_sklearn | MLPClassifier | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 12 | -2.17 |
| BTC Daily | transformer | Transformer | 276 | 112 | 164 | 40.58% | 38.33% | 40.58% | 9.42 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 276 | 104 | 172 | 37.68% | 37.08% | 37.68% | 12.32 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 286 | 106 | 180 | 37.06% | 37.08% | 37.06% | 12.94 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 276 | 96 | 180 | 34.78% | 35.00% | 34.78% | 15.22 pp | -84 | 12 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 274 | 141 | 133 | 51.46% | 51.25% | 51.46% | 1.46 pp | 8 | 22 | 0.36 |
| BTC Market Hours | transformer | Transformer | 274 | 128 | 146 | 46.72% | 46.25% | 46.72% | 3.28 pp | -18 | 22 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 274 | 127 | 147 | 46.35% | 46.25% | 46.35% | 3.65 pp | -20 | 22 | -0.91 |
| BTC Market Hours | xgb | XGBoost | 274 | 124 | 150 | 45.26% | 45.00% | 45.26% | 4.74 pp | -26 | 22 | -1.18 |
| BTC Market Hours | rf | RandomForest | 274 | 121 | 153 | 44.16% | 42.50% | 44.16% | 5.84 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 274 | 113 | 161 | 41.24% | 42.08% | 41.24% | 8.76 pp | -48 | 22 | -2.18 |

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
