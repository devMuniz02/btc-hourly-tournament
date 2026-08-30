# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T18:42:39.854882+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1150 | 862 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1026 | 661 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 17:00:00+00:00 | 635 | 423 | 211 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 17:00:00+00:00 | 636 | 476 | 158 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 76 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 76 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 0 | 76 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 05:00:00+00:00 | 76 | 0 | 76 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 423 | 208 | 215 | 49.17% | 46.25% | 49.17% | 0.83 pp | -7 | 42 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 651 | 317 | 334 | 48.69% | 46.25% | 49.79% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 651 | 315 | 336 | 48.39% | 45.83% | 49.58% | 1.61 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 423 | 199 | 224 | 47.04% | 50.42% | 47.04% | 2.96 pp | -25 | 42 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 476 | 220 | 256 | 46.22% | 47.08% | 46.22% | 3.78 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 423 | 193 | 230 | 45.63% | 41.67% | 45.63% | 4.37 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 828 | 393 | 435 | 47.46% | 47.50% | 46.67% | 2.54 pp | -42 | 44 | -0.95 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 476 | 216 | 260 | 45.38% | 44.58% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 476 | 216 | 260 | 45.38% | 45.00% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 651 | 304 | 347 | 46.70% | 42.08% | 48.75% | 3.30 pp | -43 | 40 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 828 | 388 | 440 | 46.86% | 42.50% | 46.67% | 3.14 pp | -52 | 44 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 423 | 185 | 238 | 43.74% | 43.33% | 43.74% | 6.26 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 423 | 182 | 241 | 43.03% | 42.92% | 43.03% | 6.97 pp | -59 | 42 | -1.40 |
| BTC Hourly | nn | NN | 828 | 374 | 454 | 45.17% | 42.50% | 44.58% | 4.83 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 651 | 288 | 363 | 44.24% | 41.67% | 43.54% | 5.76 pp | -75 | 40 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 476 | 197 | 279 | 41.39% | 42.50% | 41.39% | 8.61 pp | -82 | 42 | -1.95 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| BTC Hourly | rf | RandomForest | 828 | 369 | 459 | 44.57% | 43.33% | 44.17% | 5.43 pp | -90 | 44 | -2.05 |
| BTC Market Hours | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 37.50% | 39.72% | 10.28 pp | -87 | 42 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 476 | 193 | 283 | 40.55% | 38.75% | 40.55% | 9.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 651 | 278 | 373 | 42.70% | 40.83% | 43.33% | 7.30 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 828 | 359 | 469 | 43.36% | 40.42% | 43.33% | 6.64 pp | -110 | 44 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 476 | 185 | 291 | 38.87% | 35.42% | 38.87% | 11.13 pp | -106 | 42 | -2.52 |
| BTC Hourly | xgb | XGBoost | 828 | 350 | 478 | 42.27% | 39.17% | 42.50% | 7.73 pp | -128 | 44 | -2.91 |
| BTC Daily | xgb | XGBoost | 661 | 262 | 399 | 39.64% | 32.92% | 40.21% | 10.36 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 828 | 393 | 435 | 47.46% | 47.50% | 46.67% | 2.54 pp | -42 | 44 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 828 | 388 | 440 | 46.86% | 42.50% | 46.67% | 3.14 pp | -52 | 44 | -1.18 |
| BTC Hourly | nn | NN | 828 | 374 | 454 | 45.17% | 42.50% | 44.58% | 4.83 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 828 | 369 | 459 | 44.57% | 43.33% | 44.17% | 5.43 pp | -90 | 44 | -2.05 |
| BTC Hourly | lstm | LSTM | 828 | 359 | 469 | 43.36% | 40.42% | 43.33% | 6.64 pp | -110 | 44 | -2.50 |
| BTC Hourly | xgb | XGBoost | 828 | 350 | 478 | 42.27% | 39.17% | 42.50% | 7.73 pp | -128 | 44 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 651 | 317 | 334 | 48.69% | 46.25% | 49.79% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 651 | 315 | 336 | 48.39% | 45.83% | 49.58% | 1.61 pp | -21 | 40 | -0.53 |
| BTC Daily | nn | NN | 651 | 304 | 347 | 46.70% | 42.08% | 48.75% | 3.30 pp | -43 | 40 | -1.07 |
| BTC Daily | lstm | LSTM | 651 | 288 | 363 | 44.24% | 41.67% | 43.54% | 5.76 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 651 | 278 | 373 | 42.70% | 40.83% | 43.33% | 7.30 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 661 | 262 | 399 | 39.64% | 32.92% | 40.21% | 10.36 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 423 | 208 | 215 | 49.17% | 46.25% | 49.17% | 0.83 pp | -7 | 42 | -0.17 |
| BTC Market Hours | nn | NN | 423 | 199 | 224 | 47.04% | 50.42% | 47.04% | 2.96 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 423 | 193 | 230 | 45.63% | 41.67% | 45.63% | 4.37 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 423 | 185 | 238 | 43.74% | 43.33% | 43.74% | 6.26 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 423 | 182 | 241 | 43.03% | 42.92% | 43.03% | 6.97 pp | -59 | 42 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 37.50% | 39.72% | 10.28 pp | -87 | 42 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 476 | 220 | 256 | 46.22% | 47.08% | 46.22% | 3.78 pp | -36 | 42 | -0.86 |
| BTC Market Hours Daily | nn | NN | 476 | 216 | 260 | 45.38% | 44.58% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 476 | 216 | 260 | 45.38% | 45.00% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 476 | 197 | 279 | 41.39% | 42.50% | 41.39% | 8.61 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 476 | 193 | 283 | 40.55% | 38.75% | 40.55% | 9.45 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 476 | 185 | 291 | 38.87% | 35.42% | 38.87% | 11.13 pp | -106 | 42 | -2.52 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
