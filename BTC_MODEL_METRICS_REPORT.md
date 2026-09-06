# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T03:08:02.324786+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1253 | 965 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1129 | 764 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 823 | 526 | 296 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 526 | 256 | 270 | 48.67% | 45.83% | 48.75% | 1.33 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 526 | 252 | 274 | 47.91% | 47.92% | 48.54% | 2.09 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 754 | 367 | 387 | 48.67% | 47.92% | 48.75% | 1.33 pp | -20 | 44 | -0.45 |
| BTC Market Hours Daily | transformer | Transformer | 580 | 275 | 305 | 47.41% | 50.42% | 48.75% | 2.59 pp | -30 | 50 | -0.60 |
| BTC Market Hours | nn | NN | 526 | 248 | 278 | 47.15% | 50.83% | 48.54% | 2.85 pp | -30 | 50 | -0.60 |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 580 | 270 | 310 | 46.55% | 45.83% | 47.71% | 3.45 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 931 | 445 | 486 | 47.80% | 50.00% | 47.08% | 2.20 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 580 | 268 | 312 | 46.21% | 50.83% | 46.88% | 3.79 pp | -44 | 50 | -0.88 |
| BTC Daily | transformer | Transformer | 754 | 357 | 397 | 47.35% | 44.17% | 48.96% | 2.65 pp | -40 | 44 | -0.91 |
| Consolidated Hourly | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 80 | 93 | 46.24% | 46.24% | 46.24% | 3.76 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 931 | 437 | 494 | 46.94% | 45.83% | 45.21% | 3.06 pp | -57 | 49 | -1.16 |
| BTC Daily | nn | NN | 754 | 351 | 403 | 46.55% | 45.00% | 46.88% | 3.45 pp | -52 | 44 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 526 | 228 | 298 | 43.35% | 42.50% | 43.96% | 6.65 pp | -70 | 50 | -1.40 |
| BTC Market Hours | rf | RandomForest | 526 | 226 | 300 | 42.97% | 44.58% | 43.75% | 7.03 pp | -74 | 50 | -1.48 |
| Consolidated Hourly | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 76 | 97 | 43.93% | 43.93% | 43.93% | 6.07 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 526 | 217 | 309 | 41.25% | 42.92% | 41.88% | 8.75 pp | -92 | 50 | -1.84 |
| Consolidated Hourly | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 75 | 98 | 43.35% | 43.35% | 43.35% | 6.65 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 580 | 240 | 340 | 41.38% | 42.92% | 41.04% | 8.62 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 931 | 415 | 516 | 44.58% | 44.17% | 44.38% | 5.42 pp | -101 | 49 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 580 | 236 | 344 | 40.69% | 40.83% | 40.62% | 9.31 pp | -108 | 50 | -2.16 |
| BTC Hourly | nn | NN | 931 | 412 | 519 | 44.25% | 42.08% | 41.88% | 5.75 pp | -107 | 49 | -2.18 |
| Consolidated Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 580 | 231 | 349 | 39.83% | 41.25% | 39.17% | 10.17 pp | -118 | 50 | -2.36 |
| BTC Daily | lstm | LSTM | 754 | 320 | 434 | 42.44% | 35.83% | 40.62% | 7.56 pp | -114 | 44 | -2.59 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| BTC Daily | rf | RandomForest | 754 | 316 | 438 | 41.91% | 38.33% | 41.88% | 8.09 pp | -122 | 44 | -2.77 |
| BTC Hourly | lstm | LSTM | 931 | 397 | 534 | 42.64% | 37.92% | 41.46% | 7.36 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 931 | 391 | 540 | 42.00% | 40.42% | 41.04% | 8.00 pp | -149 | 49 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 764 | 300 | 464 | 39.27% | 35.42% | 37.08% | 10.73 pp | -164 | 44 | -3.73 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 931 | 445 | 486 | 47.80% | 50.00% | 47.08% | 2.20 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 931 | 437 | 494 | 46.94% | 45.83% | 45.21% | 3.06 pp | -57 | 49 | -1.16 |
| BTC Hourly | rf | RandomForest | 931 | 415 | 516 | 44.58% | 44.17% | 44.38% | 5.42 pp | -101 | 49 | -2.06 |
| BTC Hourly | nn | NN | 931 | 412 | 519 | 44.25% | 42.08% | 41.88% | 5.75 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 931 | 397 | 534 | 42.64% | 37.92% | 41.46% | 7.36 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 931 | 391 | 540 | 42.00% | 40.42% | 41.04% | 8.00 pp | -149 | 49 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 754 | 367 | 387 | 48.67% | 47.92% | 48.75% | 1.33 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 754 | 357 | 397 | 47.35% | 44.17% | 48.96% | 2.65 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 754 | 351 | 403 | 46.55% | 45.00% | 46.88% | 3.45 pp | -52 | 44 | -1.18 |
| BTC Daily | lstm | LSTM | 754 | 320 | 434 | 42.44% | 35.83% | 40.62% | 7.56 pp | -114 | 44 | -2.59 |
| BTC Daily | rf | RandomForest | 754 | 316 | 438 | 41.91% | 38.33% | 41.88% | 8.09 pp | -122 | 44 | -2.77 |
| BTC Daily | xgb | XGBoost | 764 | 300 | 464 | 39.27% | 35.42% | 37.08% | 10.73 pp | -164 | 44 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 526 | 256 | 270 | 48.67% | 45.83% | 48.75% | 1.33 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 526 | 252 | 274 | 47.91% | 47.92% | 48.54% | 2.09 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 526 | 248 | 278 | 47.15% | 50.83% | 48.54% | 2.85 pp | -30 | 50 | -0.60 |
| BTC Market Hours | lstm | LSTM | 526 | 228 | 298 | 43.35% | 42.50% | 43.96% | 6.65 pp | -70 | 50 | -1.40 |
| BTC Market Hours | rf | RandomForest | 526 | 226 | 300 | 42.97% | 44.58% | 43.75% | 7.03 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 526 | 217 | 309 | 41.25% | 42.92% | 41.88% | 8.75 pp | -92 | 50 | -1.84 |

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
