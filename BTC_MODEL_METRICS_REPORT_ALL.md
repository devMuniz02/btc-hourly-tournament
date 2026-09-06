# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T10:00:52.390827+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1258 | 970 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1134 | 769 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 828 | 531 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 830 | 585 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T22:00:00+00:00 | 177 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T22:00:00+00:00 | 177 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T22:00:00+00:00 | 177 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T22:00:00+00:00 | 178 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 531 | 258 | 273 | 48.59% | 45.42% | 48.75% | 1.41 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 531 | 254 | 277 | 47.83% | 48.33% | 48.54% | 2.17 pp | -23 | 50 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 759 | 369 | 390 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 44 | -0.48 |
| BTC Market Hours | nn | NN | 531 | 252 | 279 | 47.46% | 51.25% | 49.17% | 2.54 pp | -27 | 50 | -0.54 |
| BTC Market Hours Daily | transformer | Transformer | 585 | 278 | 307 | 47.52% | 50.83% | 48.75% | 2.48 pp | -29 | 50 | -0.58 |
| Consolidated Hourly | rf | RandomForest | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 12 | -0.58 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 12 | -0.58 |
| BTC Market Hours Daily | nn | NN | 585 | 273 | 312 | 46.67% | 46.67% | 48.12% | 3.33 pp | -39 | 50 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 936 | 447 | 489 | 47.76% | 50.00% | 47.29% | 2.24 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 585 | 271 | 314 | 46.32% | 51.25% | 47.08% | 3.68 pp | -43 | 50 | -0.86 |
| BTC Daily | transformer | Transformer | 759 | 357 | 402 | 47.04% | 42.92% | 48.12% | 2.96 pp | -45 | 44 | -1.02 |
| BTC Hourly | transformer | Transformer | 936 | 439 | 497 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| Consolidated Hourly | xgb | XGBoost | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| BTC Daily | nn | NN | 759 | 351 | 408 | 46.25% | 44.58% | 46.04% | 3.75 pp | -57 | 44 | -1.30 |
| Consolidated Hourly | lstm | LSTM | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 12 | -1.42 |
| BTC Market Hours | lstm | LSTM | 531 | 230 | 301 | 43.31% | 42.50% | 44.38% | 6.69 pp | -71 | 50 | -1.42 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| BTC Market Hours | rf | RandomForest | 531 | 228 | 303 | 42.94% | 43.75% | 43.75% | 7.06 pp | -75 | 50 | -1.50 |
| Consolidated Hourly | nn | NN | 177 | 79 | 98 | 44.63% | 44.63% | 44.63% | 5.37 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 177 | 79 | 98 | 44.63% | 44.63% | 44.63% | 5.37 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 531 | 220 | 311 | 41.43% | 42.92% | 42.08% | 8.57 pp | -91 | 50 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 585 | 243 | 342 | 41.54% | 44.17% | 41.25% | 8.46 pp | -99 | 50 | -1.98 |
| BTC Hourly | rf | RandomForest | 936 | 417 | 519 | 44.55% | 44.17% | 44.17% | 5.45 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 936 | 415 | 521 | 44.34% | 42.50% | 42.29% | 5.66 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 585 | 237 | 348 | 40.51% | 39.58% | 40.42% | 9.49 pp | -111 | 50 | -2.22 |
| Consolidated Hourly | transformer | Transformer | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 585 | 232 | 353 | 39.66% | 40.83% | 38.75% | 10.34 pp | -121 | 50 | -2.42 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 759 | 321 | 438 | 42.29% | 35.83% | 40.42% | 7.71 pp | -117 | 44 | -2.66 |
| Consolidated Market Hours | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 936 | 400 | 536 | 42.74% | 37.08% | 41.88% | 7.26 pp | -136 | 49 | -2.78 |
| BTC Daily | rf | RandomForest | 759 | 316 | 443 | 41.63% | 37.50% | 41.67% | 8.37 pp | -127 | 44 | -2.89 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 936 | 394 | 542 | 42.09% | 41.25% | 40.83% | 7.91 pp | -148 | 49 | -3.02 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 769 | 302 | 467 | 39.27% | 35.42% | 36.88% | 10.73 pp | -165 | 44 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 936 | 447 | 489 | 47.76% | 50.00% | 47.29% | 2.24 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 936 | 439 | 497 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 936 | 417 | 519 | 44.55% | 44.17% | 44.17% | 5.45 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 936 | 415 | 521 | 44.34% | 42.50% | 42.29% | 5.66 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 936 | 400 | 536 | 42.74% | 37.08% | 41.88% | 7.26 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 936 | 394 | 542 | 42.09% | 41.25% | 40.83% | 7.91 pp | -148 | 49 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 759 | 369 | 390 | 48.62% | 47.92% | 48.75% | 1.38 pp | -21 | 44 | -0.48 |
| BTC Daily | transformer | Transformer | 759 | 357 | 402 | 47.04% | 42.92% | 48.12% | 2.96 pp | -45 | 44 | -1.02 |
| BTC Daily | nn | NN | 759 | 351 | 408 | 46.25% | 44.58% | 46.04% | 3.75 pp | -57 | 44 | -1.30 |
| BTC Daily | lstm | LSTM | 759 | 321 | 438 | 42.29% | 35.83% | 40.42% | 7.71 pp | -117 | 44 | -2.66 |
| BTC Daily | rf | RandomForest | 759 | 316 | 443 | 41.63% | 37.50% | 41.67% | 8.37 pp | -127 | 44 | -2.89 |
| BTC Daily | xgb | XGBoost | 769 | 302 | 467 | 39.27% | 35.42% | 36.88% | 10.73 pp | -165 | 44 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 531 | 258 | 273 | 48.59% | 45.42% | 48.75% | 1.41 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 531 | 254 | 277 | 47.83% | 48.33% | 48.54% | 2.17 pp | -23 | 50 | -0.46 |
| BTC Market Hours | nn | NN | 531 | 252 | 279 | 47.46% | 51.25% | 49.17% | 2.54 pp | -27 | 50 | -0.54 |
| BTC Market Hours | lstm | LSTM | 531 | 230 | 301 | 43.31% | 42.50% | 44.38% | 6.69 pp | -71 | 50 | -1.42 |
| BTC Market Hours | rf | RandomForest | 531 | 228 | 303 | 42.94% | 43.75% | 43.75% | 7.06 pp | -75 | 50 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 531 | 220 | 311 | 41.43% | 42.92% | 42.08% | 8.57 pp | -91 | 50 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 585 | 278 | 307 | 47.52% | 50.83% | 48.75% | 2.48 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 585 | 273 | 312 | 46.67% | 46.67% | 48.12% | 3.33 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 585 | 271 | 314 | 46.32% | 51.25% | 47.08% | 3.68 pp | -43 | 50 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 585 | 243 | 342 | 41.54% | 44.17% | 41.25% | 8.46 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 585 | 237 | 348 | 40.51% | 39.58% | 40.42% | 9.49 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 585 | 232 | 353 | 39.66% | 40.83% | 38.75% | 10.34 pp | -121 | 50 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 12 | -0.58 |
| Consolidated Hourly | xgb | XGBoost | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 177 | 79 | 98 | 44.63% | 44.63% | 44.63% | 5.37 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 177 | 87 | 90 | 49.15% | 49.15% | 49.15% | 0.85 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 177 | 85 | 92 | 48.02% | 48.02% | 48.02% | 1.98 pp | -7 | 12 | -0.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 177 | 80 | 97 | 45.20% | 45.20% | 45.20% | 4.80 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 177 | 79 | 98 | 44.63% | 44.63% | 44.63% | 5.37 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours | lstm | LSTM | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 51 | 19 | 32 | 37.25% | 37.25% | 37.25% | 12.75 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 20 | 32 | 38.46% | 38.46% | 38.46% | 11.54 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
