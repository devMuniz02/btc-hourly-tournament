# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T18:33:01.325064+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1025 | 660 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 17:00:00+00:00 | 634 | 422 | 211 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 422 | 208 | 214 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 650 | 316 | 334 | 48.62% | 45.83% | 49.79% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 650 | 314 | 336 | 48.31% | 45.42% | 49.38% | 1.69 pp | -22 | 40 | -0.55 |
| BTC Market Hours | nn | NN | 422 | 198 | 224 | 46.92% | 50.42% | 46.92% | 3.08 pp | -26 | 42 | -0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 476 | 220 | 256 | 46.22% | 47.08% | 46.22% | 3.78 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 422 | 192 | 230 | 45.50% | 41.25% | 45.50% | 4.50 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 828 | 393 | 435 | 47.46% | 47.50% | 46.67% | 2.54 pp | -42 | 44 | -0.95 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 476 | 216 | 260 | 45.38% | 44.58% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 476 | 216 | 260 | 45.38% | 45.00% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 650 | 303 | 347 | 46.62% | 41.67% | 48.75% | 3.38 pp | -44 | 40 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 828 | 388 | 440 | 46.86% | 42.50% | 46.67% | 3.14 pp | -52 | 44 | -1.18 |
| BTC Market Hours | lstm | LSTM | 422 | 185 | 237 | 43.84% | 43.75% | 43.84% | 6.16 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| BTC Market Hours | rf | RandomForest | 422 | 182 | 240 | 43.13% | 42.92% | 43.13% | 6.87 pp | -58 | 42 | -1.38 |
| BTC Hourly | nn | NN | 828 | 374 | 454 | 45.17% | 42.50% | 44.58% | 4.83 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 650 | 288 | 362 | 44.31% | 42.08% | 43.75% | 5.69 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 476 | 197 | 279 | 41.39% | 42.50% | 41.39% | 8.61 pp | -82 | 42 | -1.95 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| BTC Hourly | rf | RandomForest | 828 | 369 | 459 | 44.57% | 43.33% | 44.17% | 5.43 pp | -90 | 44 | -2.05 |
| BTC Market Hours | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 37.92% | 39.81% | 10.19 pp | -86 | 42 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 476 | 193 | 283 | 40.55% | 38.75% | 40.55% | 9.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 650 | 277 | 373 | 42.62% | 40.83% | 43.12% | 7.38 pp | -96 | 40 | -2.40 |
| BTC Hourly | lstm | LSTM | 828 | 359 | 469 | 43.36% | 40.42% | 43.33% | 6.64 pp | -110 | 44 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 476 | 185 | 291 | 38.87% | 35.42% | 38.87% | 11.13 pp | -106 | 42 | -2.52 |
| BTC Hourly | xgb | XGBoost | 828 | 350 | 478 | 42.27% | 39.17% | 42.50% | 7.73 pp | -128 | 44 | -2.91 |
| BTC Daily | xgb | XGBoost | 660 | 261 | 399 | 39.55% | 32.50% | 40.00% | 10.45 pp | -138 | 40 | -3.45 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 650 | 316 | 334 | 48.62% | 45.83% | 49.79% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 650 | 314 | 336 | 48.31% | 45.42% | 49.38% | 1.69 pp | -22 | 40 | -0.55 |
| BTC Daily | nn | NN | 650 | 303 | 347 | 46.62% | 41.67% | 48.75% | 3.38 pp | -44 | 40 | -1.10 |
| BTC Daily | lstm | LSTM | 650 | 288 | 362 | 44.31% | 42.08% | 43.75% | 5.69 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 650 | 277 | 373 | 42.62% | 40.83% | 43.12% | 7.38 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 660 | 261 | 399 | 39.55% | 32.50% | 40.00% | 10.45 pp | -138 | 40 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 422 | 208 | 214 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Market Hours | nn | NN | 422 | 198 | 224 | 46.92% | 50.42% | 46.92% | 3.08 pp | -26 | 42 | -0.62 |
| BTC Market Hours | transformer | Transformer | 422 | 192 | 230 | 45.50% | 41.25% | 45.50% | 4.50 pp | -38 | 42 | -0.90 |
| BTC Market Hours | lstm | LSTM | 422 | 185 | 237 | 43.84% | 43.75% | 43.84% | 6.16 pp | -52 | 42 | -1.24 |
| BTC Market Hours | rf | RandomForest | 422 | 182 | 240 | 43.13% | 42.92% | 43.13% | 6.87 pp | -58 | 42 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 37.92% | 39.81% | 10.19 pp | -86 | 42 | -2.05 |

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
