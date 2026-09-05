# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T13:58:03.320841+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1244 | 956 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1120 | 755 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 12:00:00+00:00 | 802 | 517 | 284 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 12:00:00+00:00 | 804 | 571 | 231 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 163 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 163 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 44 | 119 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 44 | 119 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 517 | 251 | 266 | 48.55% | 45.83% | 48.54% | 1.45 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 517 | 249 | 268 | 48.16% | 47.50% | 48.75% | 1.84 pp | -19 | 49 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 745 | 362 | 383 | 48.59% | 47.50% | 48.96% | 1.41 pp | -21 | 44 | -0.48 |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 571 | 272 | 299 | 47.64% | 51.67% | 48.75% | 2.36 pp | -27 | 49 | -0.55 |
| BTC Market Hours | nn | NN | 517 | 244 | 273 | 47.20% | 50.00% | 48.33% | 2.80 pp | -29 | 49 | -0.59 |
| BTC Daily | transformer | Transformer | 745 | 355 | 390 | 47.65% | 45.42% | 49.58% | 2.35 pp | -35 | 44 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 922 | 441 | 481 | 47.83% | 49.58% | 47.50% | 2.17 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | nn | NN | 571 | 265 | 306 | 46.41% | 46.25% | 47.50% | 3.59 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 571 | 263 | 308 | 46.06% | 49.58% | 46.46% | 3.94 pp | -45 | 49 | -0.92 |
| BTC Hourly | transformer | Transformer | 922 | 436 | 486 | 47.29% | 47.50% | 46.04% | 2.71 pp | -50 | 48 | -1.04 |
| BTC Daily | nn | NN | 745 | 345 | 400 | 46.31% | 43.33% | 47.08% | 3.69 pp | -55 | 44 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| BTC Market Hours | lstm | LSTM | 517 | 226 | 291 | 43.71% | 43.33% | 44.17% | 6.29 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 517 | 224 | 293 | 43.33% | 45.42% | 43.75% | 6.67 pp | -69 | 49 | -1.41 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 517 | 214 | 303 | 41.39% | 43.33% | 42.08% | 8.61 pp | -89 | 49 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 571 | 238 | 333 | 41.68% | 42.92% | 40.83% | 8.32 pp | -95 | 49 | -1.94 |
| BTC Hourly | nn | NN | 922 | 411 | 511 | 44.58% | 43.75% | 42.50% | 5.42 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 922 | 411 | 511 | 44.58% | 44.17% | 43.96% | 5.42 pp | -100 | 48 | -2.08 |
| Consolidated Hourly | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 571 | 233 | 338 | 40.81% | 40.42% | 41.04% | 9.19 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 571 | 228 | 343 | 39.93% | 41.25% | 39.17% | 10.07 pp | -115 | 49 | -2.35 |
| BTC Daily | lstm | LSTM | 745 | 319 | 426 | 42.82% | 36.25% | 41.04% | 7.18 pp | -107 | 44 | -2.43 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 745 | 315 | 430 | 42.28% | 38.75% | 42.50% | 7.72 pp | -115 | 44 | -2.61 |
| BTC Hourly | lstm | LSTM | 922 | 394 | 528 | 42.73% | 38.33% | 41.46% | 7.27 pp | -134 | 48 | -2.79 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 922 | 386 | 536 | 41.87% | 39.58% | 40.21% | 8.13 pp | -150 | 48 | -3.12 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 755 | 298 | 457 | 39.47% | 36.67% | 37.50% | 10.53 pp | -159 | 44 | -3.61 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 922 | 441 | 481 | 47.83% | 49.58% | 47.50% | 2.17 pp | -40 | 48 | -0.83 |
| BTC Hourly | transformer | Transformer | 922 | 436 | 486 | 47.29% | 47.50% | 46.04% | 2.71 pp | -50 | 48 | -1.04 |
| BTC Hourly | nn | NN | 922 | 411 | 511 | 44.58% | 43.75% | 42.50% | 5.42 pp | -100 | 48 | -2.08 |
| BTC Hourly | rf | RandomForest | 922 | 411 | 511 | 44.58% | 44.17% | 43.96% | 5.42 pp | -100 | 48 | -2.08 |
| BTC Hourly | lstm | LSTM | 922 | 394 | 528 | 42.73% | 38.33% | 41.46% | 7.27 pp | -134 | 48 | -2.79 |
| BTC Hourly | xgb | XGBoost | 922 | 386 | 536 | 41.87% | 39.58% | 40.21% | 8.13 pp | -150 | 48 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 745 | 362 | 383 | 48.59% | 47.50% | 48.96% | 1.41 pp | -21 | 44 | -0.48 |
| BTC Daily | transformer | Transformer | 745 | 355 | 390 | 47.65% | 45.42% | 49.58% | 2.35 pp | -35 | 44 | -0.80 |
| BTC Daily | nn | NN | 745 | 345 | 400 | 46.31% | 43.33% | 47.08% | 3.69 pp | -55 | 44 | -1.25 |
| BTC Daily | lstm | LSTM | 745 | 319 | 426 | 42.82% | 36.25% | 41.04% | 7.18 pp | -107 | 44 | -2.43 |
| BTC Daily | rf | RandomForest | 745 | 315 | 430 | 42.28% | 38.75% | 42.50% | 7.72 pp | -115 | 44 | -2.61 |
| BTC Daily | xgb | XGBoost | 755 | 298 | 457 | 39.47% | 36.67% | 37.50% | 10.53 pp | -159 | 44 | -3.61 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 517 | 251 | 266 | 48.55% | 45.83% | 48.54% | 1.45 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 517 | 249 | 268 | 48.16% | 47.50% | 48.75% | 1.84 pp | -19 | 49 | -0.39 |
| BTC Market Hours | nn | NN | 517 | 244 | 273 | 47.20% | 50.00% | 48.33% | 2.80 pp | -29 | 49 | -0.59 |
| BTC Market Hours | lstm | LSTM | 517 | 226 | 291 | 43.71% | 43.33% | 44.17% | 6.29 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 517 | 224 | 293 | 43.33% | 45.42% | 43.75% | 6.67 pp | -69 | 49 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 517 | 214 | 303 | 41.39% | 43.33% | 42.08% | 8.61 pp | -89 | 49 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 571 | 272 | 299 | 47.64% | 51.67% | 48.75% | 2.36 pp | -27 | 49 | -0.55 |
| BTC Market Hours Daily | nn | NN | 571 | 265 | 306 | 46.41% | 46.25% | 47.50% | 3.59 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 571 | 263 | 308 | 46.06% | 49.58% | 46.46% | 3.94 pp | -45 | 49 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 571 | 238 | 333 | 41.68% | 42.92% | 40.83% | 8.32 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 571 | 233 | 338 | 40.81% | 40.42% | 41.04% | 9.19 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 571 | 228 | 343 | 39.93% | 41.25% | 39.17% | 10.07 pp | -115 | 49 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
