# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T02:30:22.438566+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 173 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 526 | 256 | 270 | 48.67% | 45.83% | 48.75% | 1.33 pp | -14 | 50 | -0.28 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 526 | 252 | 274 | 47.91% | 47.92% | 48.54% | 2.09 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 754 | 367 | 387 | 48.67% | 47.92% | 48.75% | 1.33 pp | -20 | 44 | -0.45 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 580 | 275 | 305 | 47.41% | 50.42% | 48.75% | 2.59 pp | -30 | 50 | -0.60 |
| BTC Market Hours | nn | NN | 526 | 248 | 278 | 47.15% | 50.83% | 48.54% | 2.85 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 580 | 270 | 310 | 46.55% | 45.83% | 47.71% | 3.45 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 931 | 444 | 487 | 47.69% | 49.58% | 46.88% | 2.31 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 580 | 268 | 312 | 46.21% | 50.83% | 46.88% | 3.79 pp | -44 | 50 | -0.88 |
| BTC Daily | transformer | Transformer | 754 | 357 | 397 | 47.35% | 44.17% | 48.96% | 2.65 pp | -40 | 44 | -0.91 |
| Consolidated Hourly | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| BTC Daily | nn | NN | 754 | 351 | 403 | 46.55% | 45.00% | 46.88% | 3.45 pp | -52 | 44 | -1.18 |
| BTC Hourly | transformer | Transformer | 931 | 436 | 495 | 46.83% | 45.42% | 45.00% | 3.17 pp | -59 | 49 | -1.20 |
| BTC Market Hours | lstm | LSTM | 526 | 228 | 298 | 43.35% | 42.50% | 43.96% | 6.65 pp | -70 | 50 | -1.40 |
| BTC Market Hours | rf | RandomForest | 526 | 226 | 300 | 42.97% | 44.58% | 43.75% | 7.03 pp | -74 | 50 | -1.48 |
| Consolidated Hourly | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 526 | 217 | 309 | 41.25% | 42.92% | 41.88% | 8.75 pp | -92 | 50 | -1.84 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 580 | 240 | 340 | 41.38% | 42.92% | 41.04% | 8.62 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 931 | 414 | 517 | 44.47% | 43.75% | 44.17% | 5.53 pp | -103 | 49 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 580 | 236 | 344 | 40.69% | 40.83% | 40.62% | 9.31 pp | -108 | 50 | -2.16 |
| Consolidated Hourly | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |
| BTC Hourly | nn | NN | 931 | 411 | 520 | 44.15% | 41.67% | 41.67% | 5.85 pp | -109 | 49 | -2.22 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 580 | 231 | 349 | 39.83% | 41.25% | 39.17% | 10.17 pp | -118 | 50 | -2.36 |
| BTC Daily | lstm | LSTM | 754 | 320 | 434 | 42.44% | 35.83% | 40.62% | 7.56 pp | -114 | 44 | -2.59 |
| BTC Daily | rf | RandomForest | 754 | 316 | 438 | 41.91% | 38.33% | 41.88% | 8.09 pp | -122 | 44 | -2.77 |
| BTC Hourly | lstm | LSTM | 931 | 397 | 534 | 42.64% | 37.92% | 41.46% | 7.36 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 931 | 390 | 541 | 41.89% | 40.00% | 40.83% | 8.11 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 764 | 300 | 464 | 39.27% | 35.42% | 37.08% | 10.73 pp | -164 | 44 | -3.73 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 931 | 444 | 487 | 47.69% | 49.58% | 46.88% | 2.31 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 931 | 436 | 495 | 46.83% | 45.42% | 45.00% | 3.17 pp | -59 | 49 | -1.20 |
| BTC Hourly | rf | RandomForest | 931 | 414 | 517 | 44.47% | 43.75% | 44.17% | 5.53 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 931 | 411 | 520 | 44.15% | 41.67% | 41.67% | 5.85 pp | -109 | 49 | -2.22 |
| BTC Hourly | lstm | LSTM | 931 | 397 | 534 | 42.64% | 37.92% | 41.46% | 7.36 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 931 | 390 | 541 | 41.89% | 40.00% | 40.83% | 8.11 pp | -151 | 49 | -3.08 |

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
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 172 | 84 | 88 | 48.84% | 48.84% | 48.84% | 1.16 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 172 | 79 | 93 | 45.93% | 45.93% | 45.93% | 4.07 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 172 | 77 | 95 | 44.77% | 44.77% | 44.77% | 5.23 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 172 | 76 | 96 | 44.19% | 44.19% | 44.19% | 5.81 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 172 | 73 | 99 | 42.44% | 42.44% | 42.44% | 7.56 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
