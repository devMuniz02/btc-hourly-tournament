# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T09:08:29.098369+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1322 | 1034 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1197 | 832 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 943 | 594 | 348 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 945 | 648 | 295 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 235 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 15:00:00+00:00 | 235 | 83 | 152 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 594 | 287 | 307 | 48.32% | 46.67% | 47.29% | 1.68 pp | -20 | 55 | -0.36 |
| BTC Market Hours | nn | NN | 594 | 287 | 307 | 48.32% | 52.08% | 50.00% | 1.68 pp | -20 | 55 | -0.36 |
| Consolidated Hourly | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 235 | 113 | 122 | 48.09% | 48.09% | 48.09% | 1.91 pp | -9 | 15 | -0.60 |
| BTC Market Hours | transformer | Transformer | 594 | 279 | 315 | 46.97% | 46.67% | 46.46% | 3.03 pp | -36 | 55 | -0.65 |
| BTC Market Hours Daily | nn | NN | 648 | 305 | 343 | 47.07% | 49.17% | 48.12% | 2.93 pp | -38 | 55 | -0.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 822 | 394 | 428 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 648 | 304 | 344 | 46.91% | 48.75% | 47.50% | 3.09 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 648 | 303 | 345 | 46.76% | 48.75% | 48.12% | 3.24 pp | -42 | 55 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1000 | 473 | 527 | 47.30% | 48.33% | 45.83% | 2.70 pp | -54 | 52 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 235 | 109 | 126 | 46.38% | 46.38% | 46.38% | 3.62 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 822 | 383 | 439 | 46.59% | 45.42% | 45.21% | 3.41 pp | -56 | 47 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 235 | 108 | 127 | 45.96% | 45.96% | 45.96% | 4.04 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1000 | 466 | 534 | 46.60% | 46.25% | 45.00% | 3.40 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 822 | 380 | 442 | 46.23% | 38.33% | 45.21% | 3.77 pp | -62 | 47 | -1.32 |
| BTC Market Hours | xgb | XGBoost | 594 | 255 | 339 | 42.93% | 45.00% | 43.54% | 7.07 pp | -84 | 55 | -1.53 |
| BTC Market Hours | lstm | LSTM | 594 | 253 | 341 | 42.59% | 42.08% | 42.29% | 7.41 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 594 | 253 | 341 | 42.59% | 42.08% | 42.29% | 7.41 pp | -88 | 55 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 235 | 104 | 131 | 44.26% | 44.26% | 44.26% | 5.74 pp | -27 | 15 | -1.80 |
| Consolidated Market Hours | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | rf | RandomForest | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 648 | 268 | 380 | 41.36% | 41.67% | 40.83% | 8.64 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 648 | 268 | 380 | 41.36% | 43.33% | 40.62% | 8.64 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 648 | 265 | 383 | 40.90% | 42.08% | 40.42% | 9.10 pp | -118 | 55 | -2.15 |
| BTC Hourly | nn | NN | 1000 | 441 | 559 | 44.10% | 42.08% | 41.67% | 5.90 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1000 | 440 | 560 | 44.00% | 41.67% | 43.12% | 6.00 pp | -120 | 52 | -2.31 |
| Consolidated Market Hours | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 83 | 33 | 50 | 39.76% | 39.76% | 39.76% | 10.24 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 235 | 99 | 136 | 42.13% | 42.13% | 42.13% | 7.87 pp | -37 | 15 | -2.47 |
| BTC Daily | lstm | LSTM | 822 | 348 | 474 | 42.34% | 35.42% | 40.83% | 7.66 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 822 | 342 | 480 | 41.61% | 37.50% | 41.04% | 8.39 pp | -138 | 47 | -2.94 |
| BTC Hourly | lstm | LSTM | 1000 | 423 | 577 | 42.30% | 37.08% | 39.79% | 7.70 pp | -154 | 52 | -2.96 |
| Consolidated Market Hours | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 7 | -3.00 |
| Consolidated Hourly | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 235 | 93 | 142 | 39.57% | 39.57% | 39.57% | 10.43 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 83 | 30 | 53 | 36.14% | 36.14% | 36.14% | 13.86 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1000 | 412 | 588 | 41.20% | 35.83% | 38.75% | 8.80 pp | -176 | 52 | -3.38 |
| Consolidated Market Hours | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 832 | 328 | 504 | 39.42% | 37.50% | 36.46% | 10.58 pp | -176 | 47 | -3.74 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1000 | 473 | 527 | 47.30% | 48.33% | 45.83% | 2.70 pp | -54 | 52 | -1.04 |
| BTC Hourly | transformer | Transformer | 1000 | 466 | 534 | 46.60% | 46.25% | 45.00% | 3.40 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1000 | 441 | 559 | 44.10% | 42.08% | 41.67% | 5.90 pp | -118 | 52 | -2.27 |
| BTC Hourly | rf | RandomForest | 1000 | 440 | 560 | 44.00% | 41.67% | 43.12% | 6.00 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 1000 | 423 | 577 | 42.30% | 37.08% | 39.79% | 7.70 pp | -154 | 52 | -2.96 |
| BTC Hourly | xgb | XGBoost | 1000 | 412 | 588 | 41.20% | 35.83% | 38.75% | 8.80 pp | -176 | 52 | -3.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 822 | 394 | 428 | 47.93% | 45.00% | 46.88% | 2.07 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 822 | 383 | 439 | 46.59% | 45.42% | 45.21% | 3.41 pp | -56 | 47 | -1.19 |
| BTC Daily | transformer | Transformer | 822 | 380 | 442 | 46.23% | 38.33% | 45.21% | 3.77 pp | -62 | 47 | -1.32 |
| BTC Daily | lstm | LSTM | 822 | 348 | 474 | 42.34% | 35.42% | 40.83% | 7.66 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 822 | 342 | 480 | 41.61% | 37.50% | 41.04% | 8.39 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 832 | 328 | 504 | 39.42% | 37.50% | 36.46% | 10.58 pp | -176 | 47 | -3.74 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 594 | 287 | 307 | 48.32% | 46.67% | 47.29% | 1.68 pp | -20 | 55 | -0.36 |
| BTC Market Hours | nn | NN | 594 | 287 | 307 | 48.32% | 52.08% | 50.00% | 1.68 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 594 | 279 | 315 | 46.97% | 46.67% | 46.46% | 3.03 pp | -36 | 55 | -0.65 |
| BTC Market Hours | xgb | XGBoost | 594 | 255 | 339 | 42.93% | 45.00% | 43.54% | 7.07 pp | -84 | 55 | -1.53 |
| BTC Market Hours | lstm | LSTM | 594 | 253 | 341 | 42.59% | 42.08% | 42.29% | 7.41 pp | -88 | 55 | -1.60 |
| BTC Market Hours | rf | RandomForest | 594 | 253 | 341 | 42.59% | 42.08% | 42.29% | 7.41 pp | -88 | 55 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 648 | 305 | 343 | 47.07% | 49.17% | 48.12% | 2.93 pp | -38 | 55 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 648 | 304 | 344 | 46.91% | 48.75% | 47.50% | 3.09 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 648 | 303 | 345 | 46.76% | 48.75% | 48.12% | 3.24 pp | -42 | 55 | -0.76 |
| BTC Market Hours Daily | rf | RandomForest | 648 | 268 | 380 | 41.36% | 41.67% | 40.83% | 8.64 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 648 | 268 | 380 | 41.36% | 43.33% | 40.62% | 8.64 pp | -112 | 55 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 648 | 265 | 383 | 40.90% | 42.08% | 40.42% | 9.10 pp | -118 | 55 | -2.15 |

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
