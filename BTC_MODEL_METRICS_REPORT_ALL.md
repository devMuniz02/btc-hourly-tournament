# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T11:04:02.062457+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1323 | 1035 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1199 | 834 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 945 | 596 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 947 | 650 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 596 | 287 | 309 | 48.15% | 46.25% | 47.08% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 596 | 287 | 309 | 48.15% | 51.67% | 49.79% | 1.85 pp | -22 | 55 | -0.40 |
| Consolidated Hourly | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| BTC Market Hours | transformer | Transformer | 596 | 280 | 316 | 46.98% | 46.67% | 46.46% | 3.02 pp | -36 | 55 | -0.65 |
| BTC Daily | mlp_sklearn | MLPClassifier | 824 | 395 | 429 | 47.94% | 45.00% | 46.88% | 2.06 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | nn | NN | 650 | 305 | 345 | 46.92% | 48.33% | 47.92% | 3.08 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 650 | 304 | 346 | 46.77% | 48.33% | 47.29% | 3.23 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 650 | 303 | 347 | 46.62% | 47.92% | 47.92% | 3.38 pp | -44 | 55 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1001 | 474 | 527 | 47.35% | 48.75% | 46.04% | 2.65 pp | -53 | 52 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 824 | 384 | 440 | 46.60% | 45.42% | 45.42% | 3.40 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| BTC Daily | transformer | Transformer | 824 | 381 | 443 | 46.24% | 37.92% | 45.00% | 3.76 pp | -62 | 47 | -1.32 |
| BTC Hourly | transformer | Transformer | 1001 | 466 | 535 | 46.55% | 45.83% | 44.79% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Market Hours | lstm | LSTM | 596 | 255 | 341 | 42.79% | 42.08% | 42.50% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 596 | 255 | 341 | 42.79% | 45.00% | 43.54% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 596 | 254 | 342 | 42.62% | 42.50% | 42.08% | 7.38 pp | -88 | 55 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 650 | 269 | 381 | 41.38% | 41.67% | 41.04% | 8.62 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 650 | 268 | 382 | 41.23% | 42.92% | 40.62% | 8.77 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 650 | 265 | 385 | 40.77% | 42.08% | 40.21% | 9.23 pp | -120 | 55 | -2.18 |
| BTC Hourly | nn | NN | 1001 | 442 | 559 | 44.16% | 42.08% | 41.67% | 5.84 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1001 | 441 | 560 | 44.06% | 41.67% | 43.33% | 5.94 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| BTC Daily | lstm | LSTM | 824 | 348 | 476 | 42.23% | 35.00% | 40.62% | 7.77 pp | -128 | 47 | -2.72 |
| BTC Daily | rf | RandomForest | 824 | 343 | 481 | 41.63% | 37.50% | 41.04% | 8.37 pp | -138 | 47 | -2.94 |
| BTC Hourly | lstm | LSTM | 1001 | 424 | 577 | 42.36% | 37.08% | 40.00% | 7.64 pp | -153 | 52 | -2.94 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1001 | 412 | 589 | 41.16% | 35.42% | 38.75% | 8.84 pp | -177 | 52 | -3.40 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 834 | 329 | 505 | 39.45% | 37.08% | 36.67% | 10.55 pp | -176 | 47 | -3.74 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1001 | 474 | 527 | 47.35% | 48.75% | 46.04% | 2.65 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1001 | 466 | 535 | 46.55% | 45.83% | 44.79% | 3.45 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1001 | 442 | 559 | 44.16% | 42.08% | 41.67% | 5.84 pp | -117 | 52 | -2.25 |
| BTC Hourly | rf | RandomForest | 1001 | 441 | 560 | 44.06% | 41.67% | 43.33% | 5.94 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 1001 | 424 | 577 | 42.36% | 37.08% | 40.00% | 7.64 pp | -153 | 52 | -2.94 |
| BTC Hourly | xgb | XGBoost | 1001 | 412 | 589 | 41.16% | 35.42% | 38.75% | 8.84 pp | -177 | 52 | -3.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 824 | 395 | 429 | 47.94% | 45.00% | 46.88% | 2.06 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 824 | 384 | 440 | 46.60% | 45.42% | 45.42% | 3.40 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 824 | 381 | 443 | 46.24% | 37.92% | 45.00% | 3.76 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 824 | 348 | 476 | 42.23% | 35.00% | 40.62% | 7.77 pp | -128 | 47 | -2.72 |
| BTC Daily | rf | RandomForest | 824 | 343 | 481 | 41.63% | 37.50% | 41.04% | 8.37 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 834 | 329 | 505 | 39.45% | 37.08% | 36.67% | 10.55 pp | -176 | 47 | -3.74 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 596 | 287 | 309 | 48.15% | 46.25% | 47.08% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | nn | NN | 596 | 287 | 309 | 48.15% | 51.67% | 49.79% | 1.85 pp | -22 | 55 | -0.40 |
| BTC Market Hours | transformer | Transformer | 596 | 280 | 316 | 46.98% | 46.67% | 46.46% | 3.02 pp | -36 | 55 | -0.65 |
| BTC Market Hours | lstm | LSTM | 596 | 255 | 341 | 42.79% | 42.08% | 42.50% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 596 | 255 | 341 | 42.79% | 45.00% | 43.54% | 7.21 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 596 | 254 | 342 | 42.62% | 42.50% | 42.08% | 7.38 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 650 | 305 | 345 | 46.92% | 48.33% | 47.92% | 3.08 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 650 | 304 | 346 | 46.77% | 48.33% | 47.29% | 3.23 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 650 | 303 | 347 | 46.62% | 47.92% | 47.92% | 3.38 pp | -44 | 55 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 650 | 269 | 381 | 41.38% | 41.67% | 41.04% | 8.62 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 650 | 268 | 382 | 41.23% | 42.92% | 40.62% | 8.77 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 650 | 265 | 385 | 40.77% | 42.08% | 40.21% | 9.23 pp | -120 | 55 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Hourly | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
