# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T08:52:21.053021+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1225 | 937 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1100 | 735 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 768 | 497 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 770 | 551 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 497 | 241 | 256 | 48.49% | 45.00% | 48.33% | 1.51 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 497 | 235 | 262 | 47.28% | 50.42% | 47.92% | 2.72 pp | -27 | 48 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 725 | 349 | 376 | 48.14% | 45.83% | 47.71% | 1.86 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 497 | 233 | 264 | 46.88% | 44.17% | 47.71% | 3.12 pp | -31 | 48 | -0.65 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 551 | 257 | 294 | 46.64% | 49.17% | 47.71% | 3.36 pp | -37 | 48 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Daily | transformer | Transformer | 725 | 345 | 380 | 47.59% | 46.25% | 49.58% | 2.41 pp | -35 | 43 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 551 | 254 | 297 | 46.10% | 49.17% | 46.88% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 551 | 254 | 297 | 46.10% | 45.00% | 47.50% | 3.90 pp | -43 | 48 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Daily | nn | NN | 725 | 335 | 390 | 46.21% | 44.17% | 47.29% | 3.79 pp | -55 | 43 | -1.28 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 497 | 214 | 283 | 43.06% | 40.83% | 43.12% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 497 | 214 | 283 | 43.06% | 43.75% | 43.33% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 497 | 204 | 293 | 41.05% | 41.25% | 41.25% | 8.95 pp | -89 | 48 | -1.85 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 551 | 229 | 322 | 41.56% | 42.50% | 41.04% | 8.44 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 551 | 222 | 329 | 40.29% | 38.75% | 40.62% | 9.71 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 551 | 220 | 331 | 39.93% | 41.25% | 39.17% | 10.07 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 725 | 312 | 413 | 43.03% | 37.08% | 41.46% | 6.97 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 725 | 309 | 416 | 42.62% | 40.42% | 43.33% | 7.38 pp | -107 | 43 | -2.49 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 735 | 291 | 444 | 39.59% | 36.67% | 38.54% | 10.41 pp | -153 | 43 | -3.56 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 903 | 432 | 471 | 47.84% | 50.83% | 48.33% | 2.16 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 903 | 427 | 476 | 47.29% | 47.50% | 46.88% | 2.71 pp | -49 | 48 | -1.02 |
| BTC Hourly | nn | NN | 903 | 402 | 501 | 44.52% | 44.17% | 42.08% | 5.48 pp | -99 | 48 | -2.06 |
| BTC Hourly | rf | RandomForest | 903 | 401 | 502 | 44.41% | 43.75% | 43.96% | 5.59 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 903 | 387 | 516 | 42.86% | 39.58% | 42.08% | 7.14 pp | -129 | 48 | -2.69 |
| BTC Hourly | xgb | XGBoost | 903 | 379 | 524 | 41.97% | 41.25% | 41.25% | 8.03 pp | -145 | 48 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 725 | 349 | 376 | 48.14% | 45.83% | 47.71% | 1.86 pp | -27 | 43 | -0.63 |
| BTC Daily | transformer | Transformer | 725 | 345 | 380 | 47.59% | 46.25% | 49.58% | 2.41 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 725 | 335 | 390 | 46.21% | 44.17% | 47.29% | 3.79 pp | -55 | 43 | -1.28 |
| BTC Daily | lstm | LSTM | 725 | 312 | 413 | 43.03% | 37.08% | 41.46% | 6.97 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 725 | 309 | 416 | 42.62% | 40.42% | 43.33% | 7.38 pp | -107 | 43 | -2.49 |
| BTC Daily | xgb | XGBoost | 735 | 291 | 444 | 39.59% | 36.67% | 38.54% | 10.41 pp | -153 | 43 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 497 | 241 | 256 | 48.49% | 45.00% | 48.33% | 1.51 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 497 | 235 | 262 | 47.28% | 50.42% | 47.92% | 2.72 pp | -27 | 48 | -0.56 |
| BTC Market Hours | transformer | Transformer | 497 | 233 | 264 | 46.88% | 44.17% | 47.71% | 3.12 pp | -31 | 48 | -0.65 |
| BTC Market Hours | lstm | LSTM | 497 | 214 | 283 | 43.06% | 40.83% | 43.12% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 497 | 214 | 283 | 43.06% | 43.75% | 43.33% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 497 | 204 | 293 | 41.05% | 41.25% | 41.25% | 8.95 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 551 | 257 | 294 | 46.64% | 49.17% | 47.71% | 3.36 pp | -37 | 48 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 551 | 254 | 297 | 46.10% | 49.17% | 46.88% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 551 | 254 | 297 | 46.10% | 45.00% | 47.50% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 551 | 229 | 322 | 41.56% | 42.50% | 41.04% | 8.44 pp | -93 | 48 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 551 | 222 | 329 | 40.29% | 38.75% | 40.62% | 9.71 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 551 | 220 | 331 | 39.93% | 41.25% | 39.17% | 10.07 pp | -111 | 48 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
