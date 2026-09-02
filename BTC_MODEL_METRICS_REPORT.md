# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T21:17:47.357817+00:00
Scope: `all`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1201 | 913 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1076 | 711 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 20:00:00+00:00 | 727 | 473 | 253 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 20:00:00+00:00 | 729 | 527 | 200 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 123 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 123 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 22 | 101 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 19:00:00+00:00 | 123 | 22 | 101 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 473 | 228 | 245 | 48.20% | 43.33% | 48.20% | 1.80 pp | -17 | 46 | -0.37 |
| BTC Daily | mlp_sklearn | MLPClassifier | 701 | 342 | 359 | 48.79% | 46.67% | 48.96% | 1.21 pp | -17 | 42 | -0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| BTC Market Hours | nn | NN | 473 | 223 | 250 | 47.15% | 47.92% | 47.15% | 2.85 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 701 | 337 | 364 | 48.07% | 47.08% | 49.79% | 1.93 pp | -27 | 42 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 473 | 219 | 254 | 46.30% | 40.83% | 46.30% | 3.70 pp | -35 | 46 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 527 | 242 | 285 | 45.92% | 47.92% | 46.46% | 4.08 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 527 | 241 | 286 | 45.73% | 46.67% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 527 | 241 | 286 | 45.73% | 43.33% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 879 | 416 | 463 | 47.33% | 48.33% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | transformer | Transformer | 879 | 416 | 463 | 47.33% | 48.75% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Daily | nn | NN | 701 | 325 | 376 | 46.36% | 42.92% | 48.33% | 3.64 pp | -51 | 42 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 473 | 204 | 269 | 43.13% | 42.50% | 43.13% | 6.87 pp | -65 | 46 | -1.41 |
| BTC Market Hours | lstm | LSTM | 473 | 203 | 270 | 42.92% | 40.83% | 42.92% | 7.08 pp | -67 | 46 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 473 | 193 | 280 | 40.80% | 40.00% | 40.80% | 9.20 pp | -87 | 46 | -1.89 |
| BTC Hourly | nn | NN | 879 | 395 | 484 | 44.94% | 45.83% | 43.96% | 5.06 pp | -89 | 47 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 527 | 218 | 309 | 41.37% | 41.67% | 41.46% | 8.63 pp | -91 | 46 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 879 | 391 | 488 | 44.48% | 44.58% | 44.38% | 5.52 pp | -97 | 47 | -2.06 |
| BTC Daily | lstm | LSTM | 701 | 305 | 396 | 43.51% | 38.75% | 42.50% | 6.49 pp | -91 | 42 | -2.17 |
| Consolidated Hourly | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 527 | 210 | 317 | 39.85% | 37.08% | 40.62% | 10.15 pp | -107 | 46 | -2.33 |
| BTC Daily | rf | RandomForest | 701 | 301 | 400 | 42.94% | 41.25% | 43.54% | 7.06 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 527 | 209 | 318 | 39.66% | 37.50% | 39.17% | 10.34 pp | -109 | 46 | -2.37 |
| BTC Hourly | lstm | LSTM | 879 | 375 | 504 | 42.66% | 38.33% | 42.08% | 7.34 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 879 | 372 | 507 | 42.32% | 41.67% | 42.92% | 7.68 pp | -135 | 47 | -2.87 |
| BTC Daily | xgb | XGBoost | 711 | 281 | 430 | 39.52% | 35.00% | 39.38% | 10.48 pp | -149 | 42 | -3.55 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 879 | 416 | 463 | 47.33% | 48.33% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | transformer | Transformer | 879 | 416 | 463 | 47.33% | 48.75% | 47.92% | 2.67 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 879 | 395 | 484 | 44.94% | 45.83% | 43.96% | 5.06 pp | -89 | 47 | -1.89 |
| BTC Hourly | rf | RandomForest | 879 | 391 | 488 | 44.48% | 44.58% | 44.38% | 5.52 pp | -97 | 47 | -2.06 |
| BTC Hourly | lstm | LSTM | 879 | 375 | 504 | 42.66% | 38.33% | 42.08% | 7.34 pp | -129 | 47 | -2.74 |
| BTC Hourly | xgb | XGBoost | 879 | 372 | 507 | 42.32% | 41.67% | 42.92% | 7.68 pp | -135 | 47 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 701 | 342 | 359 | 48.79% | 46.67% | 48.96% | 1.21 pp | -17 | 42 | -0.40 |
| BTC Daily | transformer | Transformer | 701 | 337 | 364 | 48.07% | 47.08% | 49.79% | 1.93 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 701 | 325 | 376 | 46.36% | 42.92% | 48.33% | 3.64 pp | -51 | 42 | -1.21 |
| BTC Daily | lstm | LSTM | 701 | 305 | 396 | 43.51% | 38.75% | 42.50% | 6.49 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 701 | 301 | 400 | 42.94% | 41.25% | 43.54% | 7.06 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 711 | 281 | 430 | 39.52% | 35.00% | 39.38% | 10.48 pp | -149 | 42 | -3.55 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 473 | 228 | 245 | 48.20% | 43.33% | 48.20% | 1.80 pp | -17 | 46 | -0.37 |
| BTC Market Hours | nn | NN | 473 | 223 | 250 | 47.15% | 47.92% | 47.15% | 2.85 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 473 | 219 | 254 | 46.30% | 40.83% | 46.30% | 3.70 pp | -35 | 46 | -0.76 |
| BTC Market Hours | rf | RandomForest | 473 | 204 | 269 | 43.13% | 42.50% | 43.13% | 6.87 pp | -65 | 46 | -1.41 |
| BTC Market Hours | lstm | LSTM | 473 | 203 | 270 | 42.92% | 40.83% | 42.92% | 7.08 pp | -67 | 46 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 473 | 193 | 280 | 40.80% | 40.00% | 40.80% | 9.20 pp | -87 | 46 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 527 | 242 | 285 | 45.92% | 47.92% | 46.46% | 4.08 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 527 | 241 | 286 | 45.73% | 46.67% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 527 | 241 | 286 | 45.73% | 43.33% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 527 | 218 | 309 | 41.37% | 41.67% | 41.46% | 8.63 pp | -91 | 46 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 527 | 210 | 317 | 39.85% | 37.08% | 40.62% | 10.15 pp | -107 | 46 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 527 | 209 | 318 | 39.66% | 37.50% | 39.17% | 10.34 pp | -109 | 46 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 64 | 59 | 52.03% | 52.03% | 52.03% | 2.03 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 58 | 65 | 47.15% | 47.15% | 47.15% | 2.85 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 55 | 68 | 44.72% | 44.72% | 44.72% | 5.28 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 50 | 73 | 40.65% | 40.65% | 40.65% | 9.35 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
