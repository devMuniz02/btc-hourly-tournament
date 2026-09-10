# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T05:23:19.288739+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1319 | 1031 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1195 | 830 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 941 | 592 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 943 | 646 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 232 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 232 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 232 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T13:00:00+00:00 | 233 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 592 | 287 | 305 | 48.48% | 47.50% | 47.29% | 1.52 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 592 | 286 | 306 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 592 | 278 | 314 | 46.96% | 46.67% | 46.25% | 3.04 pp | -36 | 55 | -0.65 |
| Consolidated Hourly | rf | RandomForest | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 15 | -0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 15 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 820 | 393 | 427 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 646 | 303 | 343 | 46.90% | 48.75% | 47.29% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 646 | 303 | 343 | 46.90% | 48.33% | 48.12% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 646 | 301 | 345 | 46.59% | 48.33% | 47.71% | 3.41 pp | -44 | 55 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 997 | 473 | 524 | 47.44% | 49.17% | 46.25% | 2.56 pp | -51 | 52 | -0.98 |
| BTC Daily | nn | NN | 820 | 382 | 438 | 46.59% | 45.00% | 45.42% | 3.41 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| BTC Daily | transformer | Transformer | 820 | 380 | 440 | 46.34% | 39.17% | 45.62% | 3.66 pp | -60 | 47 | -1.28 |
| BTC Hourly | transformer | Transformer | 997 | 463 | 534 | 46.44% | 45.00% | 44.58% | 3.56 pp | -71 | 52 | -1.37 |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 592 | 254 | 338 | 42.91% | 45.00% | 43.54% | 7.09 pp | -84 | 55 | -1.53 |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | xgb | XGBoost | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 15 | -1.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 15 | -1.73 |
| BTC Market Hours Daily | rf | RandomForest | 646 | 267 | 379 | 41.33% | 41.67% | 40.62% | 8.67 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 646 | 266 | 380 | 41.18% | 42.92% | 40.42% | 8.82 pp | -114 | 55 | -2.07 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 646 | 263 | 383 | 40.71% | 41.67% | 40.00% | 9.29 pp | -120 | 55 | -2.18 |
| BTC Hourly | nn | NN | 997 | 440 | 557 | 44.13% | 42.50% | 41.88% | 5.87 pp | -117 | 52 | -2.25 |
| Consolidated Hourly | transformer | Transformer | 232 | 99 | 133 | 42.67% | 42.67% | 42.67% | 7.33 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 232 | 99 | 133 | 42.67% | 42.67% | 42.67% | 7.33 pp | -34 | 15 | -2.27 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| BTC Hourly | rf | RandomForest | 997 | 439 | 558 | 44.03% | 41.25% | 43.12% | 5.97 pp | -119 | 52 | -2.29 |
| BTC Daily | lstm | LSTM | 820 | 348 | 472 | 42.44% | 36.67% | 41.04% | 7.56 pp | -124 | 47 | -2.64 |
| Consolidated Hourly | nn | NN | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 15 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 15 | -2.67 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 997 | 423 | 574 | 42.43% | 37.50% | 40.21% | 7.57 pp | -151 | 52 | -2.90 |
| BTC Daily | rf | RandomForest | 820 | 341 | 479 | 41.59% | 37.92% | 41.04% | 8.41 pp | -138 | 47 | -2.94 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 997 | 410 | 587 | 41.12% | 35.00% | 38.54% | 8.88 pp | -177 | 52 | -3.40 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 830 | 327 | 503 | 39.40% | 37.50% | 36.25% | 10.60 pp | -176 | 47 | -3.74 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 997 | 473 | 524 | 47.44% | 49.17% | 46.25% | 2.56 pp | -51 | 52 | -0.98 |
| BTC Hourly | transformer | Transformer | 997 | 463 | 534 | 46.44% | 45.00% | 44.58% | 3.56 pp | -71 | 52 | -1.37 |
| BTC Hourly | nn | NN | 997 | 440 | 557 | 44.13% | 42.50% | 41.88% | 5.87 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 997 | 439 | 558 | 44.03% | 41.25% | 43.12% | 5.97 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 997 | 423 | 574 | 42.43% | 37.50% | 40.21% | 7.57 pp | -151 | 52 | -2.90 |
| BTC Hourly | xgb | XGBoost | 997 | 410 | 587 | 41.12% | 35.00% | 38.54% | 8.88 pp | -177 | 52 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 820 | 393 | 427 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 820 | 382 | 438 | 46.59% | 45.00% | 45.42% | 3.41 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 820 | 380 | 440 | 46.34% | 39.17% | 45.62% | 3.66 pp | -60 | 47 | -1.28 |
| BTC Daily | lstm | LSTM | 820 | 348 | 472 | 42.44% | 36.67% | 41.04% | 7.56 pp | -124 | 47 | -2.64 |
| BTC Daily | rf | RandomForest | 820 | 341 | 479 | 41.59% | 37.92% | 41.04% | 8.41 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 830 | 327 | 503 | 39.40% | 37.50% | 36.25% | 10.60 pp | -176 | 47 | -3.74 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 592 | 287 | 305 | 48.48% | 47.50% | 47.29% | 1.52 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 592 | 286 | 306 | 48.31% | 52.50% | 50.00% | 1.69 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 592 | 278 | 314 | 46.96% | 46.67% | 46.25% | 3.04 pp | -36 | 55 | -0.65 |
| BTC Market Hours | xgb | XGBoost | 592 | 254 | 338 | 42.91% | 45.00% | 43.54% | 7.09 pp | -84 | 55 | -1.53 |
| BTC Market Hours | lstm | LSTM | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 592 | 252 | 340 | 42.57% | 42.50% | 42.50% | 7.43 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 646 | 303 | 343 | 46.90% | 48.75% | 47.29% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 646 | 303 | 343 | 46.90% | 48.33% | 48.12% | 3.10 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 646 | 301 | 345 | 46.59% | 48.33% | 47.71% | 3.41 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 646 | 267 | 379 | 41.33% | 41.67% | 40.62% | 8.67 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 646 | 266 | 380 | 41.18% | 42.92% | 40.42% | 8.82 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 646 | 263 | 383 | 40.71% | 41.67% | 40.00% | 9.29 pp | -120 | 55 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 15 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 15 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 232 | 99 | 133 | 42.67% | 42.67% | 42.67% | 7.33 pp | -34 | 15 | -2.27 |
| Consolidated Hourly | nn | NN | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 15 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 232 | 111 | 121 | 47.84% | 47.84% | 47.84% | 2.16 pp | -10 | 15 | -0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 232 | 107 | 125 | 46.12% | 46.12% | 46.12% | 3.88 pp | -18 | 15 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 15 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 232 | 99 | 133 | 42.67% | 42.67% | 42.67% | 7.33 pp | -34 | 15 | -2.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 15 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | transformer | Transformer | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 81 | 30 | 51 | 37.04% | 37.04% | 37.04% | 12.96 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 82 | 36 | 46 | 43.90% | 43.90% | 43.90% | 6.10 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 82 | 35 | 47 | 42.68% | 42.68% | 42.68% | 7.32 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 82 | 30 | 52 | 36.59% | 36.59% | 36.59% | 13.41 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
