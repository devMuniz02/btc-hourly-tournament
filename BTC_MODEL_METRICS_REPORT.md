# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T20:55:48.694886+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1200 | 912 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1076 | 711 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 19:00:00+00:00 | 726 | 473 | 252 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 19:00:00+00:00 | 728 | 527 | 199 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 123 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T19:00:00+00:00 | 124 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 701 | 343 | 358 | 48.93% | 47.08% | 49.17% | 1.07 pp | -15 | 42 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 473 | 228 | 245 | 48.20% | 43.33% | 48.20% | 1.80 pp | -17 | 46 | -0.37 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 473 | 223 | 250 | 47.15% | 47.92% | 47.15% | 2.85 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 701 | 337 | 364 | 48.07% | 47.08% | 49.79% | 1.93 pp | -27 | 42 | -0.64 |
| BTC Market Hours | transformer | Transformer | 473 | 219 | 254 | 46.30% | 40.83% | 46.30% | 3.70 pp | -35 | 46 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 527 | 242 | 285 | 45.92% | 47.92% | 46.46% | 4.08 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 527 | 241 | 286 | 45.73% | 46.67% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 527 | 241 | 286 | 45.73% | 43.33% | 46.46% | 4.27 pp | -45 | 46 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 878 | 416 | 462 | 47.38% | 48.33% | 47.92% | 2.62 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours | rf | RandomForest | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Hourly | transformer | Transformer | 878 | 415 | 463 | 47.27% | 48.33% | 47.71% | 2.73 pp | -48 | 47 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 701 | 326 | 375 | 46.50% | 43.33% | 48.54% | 3.50 pp | -49 | 42 | -1.17 |
| BTC Market Hours | rf | RandomForest | 473 | 204 | 269 | 43.13% | 42.50% | 43.13% | 6.87 pp | -65 | 46 | -1.41 |
| BTC Market Hours | lstm | LSTM | 473 | 203 | 270 | 42.92% | 40.83% | 42.92% | 7.08 pp | -67 | 46 | -1.46 |
| Consolidated Hourly | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |
| BTC Hourly | nn | NN | 878 | 395 | 483 | 44.99% | 46.25% | 43.96% | 5.01 pp | -88 | 47 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 473 | 193 | 280 | 40.80% | 40.00% | 40.80% | 9.20 pp | -87 | 46 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 527 | 218 | 309 | 41.37% | 41.67% | 41.46% | 8.63 pp | -91 | 46 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 878 | 391 | 487 | 44.53% | 44.58% | 44.38% | 5.47 pp | -96 | 47 | -2.04 |
| BTC Daily | lstm | LSTM | 701 | 304 | 397 | 43.37% | 38.33% | 42.29% | 6.63 pp | -93 | 42 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 527 | 210 | 317 | 39.85% | 37.08% | 40.62% | 10.15 pp | -107 | 46 | -2.33 |
| BTC Daily | rf | RandomForest | 701 | 301 | 400 | 42.94% | 41.25% | 43.54% | 7.06 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 527 | 209 | 318 | 39.66% | 37.50% | 39.17% | 10.34 pp | -109 | 46 | -2.37 |
| BTC Hourly | lstm | LSTM | 878 | 374 | 504 | 42.60% | 37.92% | 41.88% | 7.40 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 878 | 372 | 506 | 42.37% | 41.67% | 43.12% | 7.63 pp | -134 | 47 | -2.85 |
| BTC Daily | xgb | XGBoost | 711 | 282 | 429 | 39.66% | 35.42% | 39.58% | 10.34 pp | -147 | 42 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 22 | 5 | 17 | 22.73% | 22.73% | 22.73% | 27.27 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 878 | 416 | 462 | 47.38% | 48.33% | 47.92% | 2.62 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 878 | 415 | 463 | 47.27% | 48.33% | 47.71% | 2.73 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 878 | 395 | 483 | 44.99% | 46.25% | 43.96% | 5.01 pp | -88 | 47 | -1.87 |
| BTC Hourly | rf | RandomForest | 878 | 391 | 487 | 44.53% | 44.58% | 44.38% | 5.47 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 878 | 374 | 504 | 42.60% | 37.92% | 41.88% | 7.40 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 878 | 372 | 506 | 42.37% | 41.67% | 43.12% | 7.63 pp | -134 | 47 | -2.85 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 701 | 343 | 358 | 48.93% | 47.08% | 49.17% | 1.07 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 701 | 337 | 364 | 48.07% | 47.08% | 49.79% | 1.93 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 701 | 326 | 375 | 46.50% | 43.33% | 48.54% | 3.50 pp | -49 | 42 | -1.17 |
| BTC Daily | lstm | LSTM | 701 | 304 | 397 | 43.37% | 38.33% | 42.29% | 6.63 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 701 | 301 | 400 | 42.94% | 41.25% | 43.54% | 7.06 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 711 | 282 | 429 | 39.66% | 35.42% | 39.58% | 10.34 pp | -147 | 42 | -3.50 |

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
| Consolidated Hourly | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 123 | 59 | 64 | 47.97% | 47.97% | 47.97% | 2.03 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 123 | 54 | 69 | 43.90% | 43.90% | 43.90% | 6.10 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 123 | 53 | 70 | 43.09% | 43.09% | 43.09% | 6.91 pp | -17 | 10 | -1.70 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
