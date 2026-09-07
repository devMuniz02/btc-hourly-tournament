# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T17:04:46.976537+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1279 | 991 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1155 | 790 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 16:00:00+00:00 | 867 | 552 | 314 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 16:00:00+00:00 | 869 | 606 | 261 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 198 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 552 | 268 | 284 | 48.55% | 46.67% | 47.71% | 1.45 pp | -16 | 52 | -0.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 552 | 264 | 288 | 47.83% | 51.67% | 49.79% | 2.17 pp | -24 | 52 | -0.46 |
| BTC Market Hours | transformer | Transformer | 552 | 261 | 291 | 47.28% | 47.50% | 47.92% | 2.72 pp | -30 | 52 | -0.58 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 606 | 285 | 321 | 47.03% | 49.58% | 47.92% | 2.97 pp | -36 | 52 | -0.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 780 | 374 | 406 | 47.95% | 45.42% | 47.29% | 2.05 pp | -32 | 45 | -0.71 |
| BTC Market Hours Daily | nn | NN | 606 | 282 | 324 | 46.53% | 47.50% | 48.12% | 3.47 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 606 | 280 | 326 | 46.20% | 48.33% | 46.88% | 3.80 pp | -46 | 52 | -0.88 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 957 | 456 | 501 | 47.65% | 49.58% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 780 | 362 | 418 | 46.41% | 40.83% | 46.46% | 3.59 pp | -56 | 45 | -1.24 |
| BTC Daily | nn | NN | 780 | 361 | 419 | 46.28% | 44.58% | 45.21% | 3.72 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 957 | 446 | 511 | 46.60% | 45.00% | 43.96% | 3.40 pp | -65 | 50 | -1.30 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 552 | 239 | 313 | 43.30% | 45.42% | 43.12% | 6.70 pp | -74 | 52 | -1.42 |
| Consolidated Hourly | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| BTC Market Hours | lstm | LSTM | 552 | 237 | 315 | 42.93% | 41.25% | 43.12% | 7.07 pp | -78 | 52 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 552 | 232 | 320 | 42.03% | 44.58% | 42.08% | 7.97 pp | -88 | 52 | -1.69 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 606 | 253 | 353 | 41.75% | 43.33% | 41.04% | 8.25 pp | -100 | 52 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 957 | 425 | 532 | 44.41% | 43.75% | 43.54% | 5.59 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 957 | 424 | 533 | 44.31% | 42.92% | 42.92% | 5.69 pp | -109 | 50 | -2.18 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 606 | 245 | 361 | 40.43% | 39.58% | 40.62% | 9.57 pp | -116 | 52 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 606 | 244 | 362 | 40.26% | 40.83% | 39.58% | 9.74 pp | -118 | 52 | -2.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Hourly | lstm | LSTM | 957 | 409 | 548 | 42.74% | 37.50% | 42.29% | 7.26 pp | -139 | 50 | -2.78 |
| BTC Daily | lstm | LSTM | 780 | 327 | 453 | 41.92% | 34.17% | 39.58% | 8.08 pp | -126 | 45 | -2.80 |
| BTC Daily | rf | RandomForest | 780 | 327 | 453 | 41.92% | 38.75% | 42.08% | 8.08 pp | -126 | 45 | -2.80 |
| BTC Hourly | xgb | XGBoost | 957 | 397 | 560 | 41.48% | 38.75% | 39.58% | 8.52 pp | -163 | 50 | -3.26 |
| BTC Daily | xgb | XGBoost | 790 | 307 | 483 | 38.86% | 35.00% | 36.04% | 11.14 pp | -176 | 45 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 957 | 456 | 501 | 47.65% | 49.58% | 47.29% | 2.35 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 957 | 446 | 511 | 46.60% | 45.00% | 43.96% | 3.40 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 957 | 425 | 532 | 44.41% | 43.75% | 43.54% | 5.59 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 957 | 424 | 533 | 44.31% | 42.92% | 42.92% | 5.69 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 957 | 409 | 548 | 42.74% | 37.50% | 42.29% | 7.26 pp | -139 | 50 | -2.78 |
| BTC Hourly | xgb | XGBoost | 957 | 397 | 560 | 41.48% | 38.75% | 39.58% | 8.52 pp | -163 | 50 | -3.26 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 780 | 374 | 406 | 47.95% | 45.42% | 47.29% | 2.05 pp | -32 | 45 | -0.71 |
| BTC Daily | transformer | Transformer | 780 | 362 | 418 | 46.41% | 40.83% | 46.46% | 3.59 pp | -56 | 45 | -1.24 |
| BTC Daily | nn | NN | 780 | 361 | 419 | 46.28% | 44.58% | 45.21% | 3.72 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 780 | 327 | 453 | 41.92% | 34.17% | 39.58% | 8.08 pp | -126 | 45 | -2.80 |
| BTC Daily | rf | RandomForest | 780 | 327 | 453 | 41.92% | 38.75% | 42.08% | 8.08 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 790 | 307 | 483 | 38.86% | 35.00% | 36.04% | 11.14 pp | -176 | 45 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 552 | 268 | 284 | 48.55% | 46.67% | 47.71% | 1.45 pp | -16 | 52 | -0.31 |
| BTC Market Hours | nn | NN | 552 | 264 | 288 | 47.83% | 51.67% | 49.79% | 2.17 pp | -24 | 52 | -0.46 |
| BTC Market Hours | transformer | Transformer | 552 | 261 | 291 | 47.28% | 47.50% | 47.92% | 2.72 pp | -30 | 52 | -0.58 |
| BTC Market Hours | rf | RandomForest | 552 | 239 | 313 | 43.30% | 45.42% | 43.12% | 6.70 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 552 | 237 | 315 | 42.93% | 41.25% | 43.12% | 7.07 pp | -78 | 52 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 552 | 232 | 320 | 42.03% | 44.58% | 42.08% | 7.97 pp | -88 | 52 | -1.69 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 606 | 285 | 321 | 47.03% | 49.58% | 47.92% | 2.97 pp | -36 | 52 | -0.69 |
| BTC Market Hours Daily | nn | NN | 606 | 282 | 324 | 46.53% | 47.50% | 48.12% | 3.47 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 606 | 280 | 326 | 46.20% | 48.33% | 46.88% | 3.80 pp | -46 | 52 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 606 | 253 | 353 | 41.75% | 43.33% | 41.04% | 8.25 pp | -100 | 52 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 606 | 245 | 361 | 40.43% | 39.58% | 40.62% | 9.57 pp | -116 | 52 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 606 | 244 | 362 | 40.26% | 40.83% | 39.58% | 9.74 pp | -118 | 52 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 5 | -2.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
