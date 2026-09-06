# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T03:27:02.570839+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1254 | 966 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1130 | 765 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 824 | 527 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 825 | 580 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 173 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 173 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 49 | 124 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 20:00:00+00:00 | 173 | 49 | 124 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 527 | 256 | 271 | 48.58% | 45.83% | 48.54% | 1.42 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 527 | 252 | 275 | 47.82% | 47.92% | 48.33% | 2.18 pp | -23 | 50 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 755 | 367 | 388 | 48.61% | 47.92% | 48.75% | 1.39 pp | -21 | 44 | -0.48 |
| BTC Market Hours Daily | transformer | Transformer | 580 | 275 | 305 | 47.41% | 50.42% | 48.75% | 2.59 pp | -30 | 50 | -0.60 |
| BTC Market Hours | nn | NN | 527 | 248 | 279 | 47.06% | 50.83% | 48.33% | 2.94 pp | -31 | 50 | -0.62 |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 580 | 270 | 310 | 46.55% | 45.83% | 47.71% | 3.45 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 932 | 445 | 487 | 47.75% | 49.58% | 47.08% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 580 | 268 | 312 | 46.21% | 50.83% | 46.88% | 3.79 pp | -44 | 50 | -0.88 |
| BTC Daily | transformer | Transformer | 755 | 357 | 398 | 47.28% | 43.75% | 48.75% | 2.72 pp | -41 | 44 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 932 | 437 | 495 | 46.89% | 45.83% | 45.21% | 3.11 pp | -58 | 49 | -1.18 |
| BTC Daily | nn | NN | 755 | 351 | 404 | 46.49% | 45.00% | 46.67% | 3.51 pp | -53 | 44 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 527 | 228 | 299 | 43.26% | 42.50% | 43.96% | 6.74 pp | -71 | 50 | -1.42 |
| BTC Market Hours | rf | RandomForest | 527 | 226 | 301 | 42.88% | 44.58% | 43.54% | 7.12 pp | -75 | 50 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 527 | 217 | 310 | 41.18% | 42.92% | 41.67% | 8.82 pp | -93 | 50 | -1.86 |
| Consolidated Hourly | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 580 | 240 | 340 | 41.38% | 42.92% | 41.04% | 8.62 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 932 | 415 | 517 | 44.53% | 44.17% | 44.17% | 5.47 pp | -102 | 49 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 580 | 236 | 344 | 40.69% | 40.83% | 40.62% | 9.31 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 932 | 412 | 520 | 44.21% | 42.08% | 41.88% | 5.79 pp | -108 | 49 | -2.20 |
| Consolidated Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 580 | 231 | 349 | 39.83% | 41.25% | 39.17% | 10.17 pp | -118 | 50 | -2.36 |
| BTC Daily | lstm | LSTM | 755 | 320 | 435 | 42.38% | 35.83% | 40.42% | 7.62 pp | -115 | 44 | -2.61 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| BTC Daily | rf | RandomForest | 755 | 316 | 439 | 41.85% | 38.33% | 41.88% | 8.15 pp | -123 | 44 | -2.80 |
| BTC Hourly | lstm | LSTM | 932 | 397 | 535 | 42.60% | 37.50% | 41.46% | 7.40 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 932 | 391 | 541 | 41.95% | 40.42% | 40.83% | 8.05 pp | -150 | 49 | -3.06 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 765 | 300 | 465 | 39.22% | 35.42% | 36.88% | 10.78 pp | -165 | 44 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 932 | 445 | 487 | 47.75% | 49.58% | 47.08% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 932 | 437 | 495 | 46.89% | 45.83% | 45.21% | 3.11 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 932 | 415 | 517 | 44.53% | 44.17% | 44.17% | 5.47 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 932 | 412 | 520 | 44.21% | 42.08% | 41.88% | 5.79 pp | -108 | 49 | -2.20 |
| BTC Hourly | lstm | LSTM | 932 | 397 | 535 | 42.60% | 37.50% | 41.46% | 7.40 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 932 | 391 | 541 | 41.95% | 40.42% | 40.83% | 8.05 pp | -150 | 49 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 755 | 367 | 388 | 48.61% | 47.92% | 48.75% | 1.39 pp | -21 | 44 | -0.48 |
| BTC Daily | transformer | Transformer | 755 | 357 | 398 | 47.28% | 43.75% | 48.75% | 2.72 pp | -41 | 44 | -0.93 |
| BTC Daily | nn | NN | 755 | 351 | 404 | 46.49% | 45.00% | 46.67% | 3.51 pp | -53 | 44 | -1.20 |
| BTC Daily | lstm | LSTM | 755 | 320 | 435 | 42.38% | 35.83% | 40.42% | 7.62 pp | -115 | 44 | -2.61 |
| BTC Daily | rf | RandomForest | 755 | 316 | 439 | 41.85% | 38.33% | 41.88% | 8.15 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 765 | 300 | 465 | 39.22% | 35.42% | 36.88% | 10.78 pp | -165 | 44 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 527 | 256 | 271 | 48.58% | 45.83% | 48.54% | 1.42 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 527 | 252 | 275 | 47.82% | 47.92% | 48.33% | 2.18 pp | -23 | 50 | -0.46 |
| BTC Market Hours | nn | NN | 527 | 248 | 279 | 47.06% | 50.83% | 48.33% | 2.94 pp | -31 | 50 | -0.62 |
| BTC Market Hours | lstm | LSTM | 527 | 228 | 299 | 43.26% | 42.50% | 43.96% | 6.74 pp | -71 | 50 | -1.42 |
| BTC Market Hours | rf | RandomForest | 527 | 226 | 301 | 42.88% | 44.58% | 43.54% | 7.12 pp | -75 | 50 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 527 | 217 | 310 | 41.18% | 42.92% | 41.67% | 8.82 pp | -93 | 50 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 580 | 275 | 305 | 47.41% | 50.42% | 48.75% | 2.59 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 580 | 270 | 310 | 46.55% | 45.83% | 47.71% | 3.45 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 580 | 268 | 312 | 46.21% | 50.83% | 46.88% | 3.79 pp | -44 | 50 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 580 | 240 | 340 | 41.38% | 42.92% | 41.04% | 8.62 pp | -100 | 50 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 580 | 236 | 344 | 40.69% | 40.83% | 40.62% | 9.31 pp | -108 | 50 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 580 | 231 | 349 | 39.83% | 41.25% | 39.17% | 10.17 pp | -118 | 50 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 86 | 87 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
