# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T02:05:00.174835+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 824 | 579 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 526 | 256 | 270 | 48.67% | 45.83% | 48.75% | 1.33 pp | -14 | 50 | -0.28 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| BTC Market Hours | transformer | Transformer | 526 | 252 | 274 | 47.91% | 47.92% | 48.54% | 2.09 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 754 | 367 | 387 | 48.67% | 47.92% | 48.75% | 1.33 pp | -20 | 44 | -0.45 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 579 | 275 | 304 | 47.50% | 50.42% | 48.75% | 2.50 pp | -29 | 50 | -0.58 |
| BTC Market Hours | nn | NN | 526 | 248 | 278 | 47.15% | 50.83% | 48.54% | 2.85 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 579 | 270 | 309 | 46.63% | 46.25% | 47.92% | 3.37 pp | -39 | 50 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 931 | 444 | 487 | 47.69% | 49.58% | 46.88% | 2.31 pp | -43 | 49 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 579 | 267 | 312 | 46.11% | 50.42% | 46.67% | 3.89 pp | -45 | 50 | -0.90 |
| BTC Daily | transformer | Transformer | 754 | 357 | 397 | 47.35% | 44.17% | 48.96% | 2.65 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 754 | 351 | 403 | 46.55% | 45.00% | 46.88% | 3.45 pp | -52 | 44 | -1.18 |
| BTC Hourly | transformer | Transformer | 931 | 436 | 495 | 46.83% | 45.42% | 45.00% | 3.17 pp | -59 | 49 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| BTC Market Hours | lstm | LSTM | 526 | 228 | 298 | 43.35% | 42.50% | 43.96% | 6.65 pp | -70 | 50 | -1.40 |
| BTC Market Hours | rf | RandomForest | 526 | 226 | 300 | 42.97% | 44.58% | 43.75% | 7.03 pp | -74 | 50 | -1.48 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 526 | 217 | 309 | 41.25% | 42.92% | 41.88% | 8.75 pp | -92 | 50 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 579 | 240 | 339 | 41.45% | 43.33% | 41.04% | 8.55 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 931 | 414 | 517 | 44.47% | 43.75% | 44.17% | 5.53 pp | -103 | 49 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 579 | 236 | 343 | 40.76% | 41.25% | 40.62% | 9.24 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 931 | 411 | 520 | 44.15% | 41.67% | 41.67% | 5.85 pp | -109 | 49 | -2.22 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 579 | 231 | 348 | 39.90% | 41.67% | 39.17% | 10.10 pp | -117 | 50 | -2.34 |
| BTC Daily | lstm | LSTM | 754 | 320 | 434 | 42.44% | 35.83% | 40.62% | 7.56 pp | -114 | 44 | -2.59 |
| BTC Daily | rf | RandomForest | 754 | 316 | 438 | 41.91% | 38.33% | 41.88% | 8.09 pp | -122 | 44 | -2.77 |
| BTC Hourly | lstm | LSTM | 931 | 397 | 534 | 42.64% | 37.92% | 41.46% | 7.36 pp | -137 | 49 | -2.80 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 931 | 390 | 541 | 41.89% | 40.00% | 40.83% | 8.11 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 764 | 300 | 464 | 39.27% | 35.42% | 37.08% | 10.73 pp | -164 | 44 | -3.73 |

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
| BTC Market Hours Daily | transformer | Transformer | 579 | 275 | 304 | 47.50% | 50.42% | 48.75% | 2.50 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 579 | 270 | 309 | 46.63% | 46.25% | 47.92% | 3.37 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 579 | 267 | 312 | 46.11% | 50.42% | 46.67% | 3.89 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 579 | 240 | 339 | 41.45% | 43.33% | 41.04% | 8.55 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 579 | 236 | 343 | 40.76% | 41.25% | 40.62% | 9.24 pp | -107 | 50 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 579 | 231 | 348 | 39.90% | 41.67% | 39.17% | 10.10 pp | -117 | 50 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
