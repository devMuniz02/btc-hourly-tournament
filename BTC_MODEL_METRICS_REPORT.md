# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T17:41:40.333530+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1149 | 861 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1025 | 660 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 633 | 422 | 210 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 635 | 476 | 157 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 77 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 422 | 208 | 214 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 650 | 317 | 333 | 48.77% | 46.25% | 50.00% | 1.23 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 650 | 315 | 335 | 48.46% | 45.83% | 49.58% | 1.54 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 422 | 198 | 224 | 46.92% | 50.42% | 46.92% | 3.08 pp | -26 | 42 | -0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 476 | 220 | 256 | 46.22% | 47.08% | 46.22% | 3.78 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 422 | 192 | 230 | 45.50% | 41.25% | 45.50% | 4.50 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 827 | 393 | 434 | 47.52% | 47.50% | 46.67% | 2.48 pp | -41 | 44 | -0.93 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 476 | 216 | 260 | 45.38% | 44.58% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 476 | 216 | 260 | 45.38% | 45.00% | 45.38% | 4.62 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 650 | 304 | 346 | 46.77% | 42.08% | 48.96% | 3.23 pp | -42 | 40 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 827 | 387 | 440 | 46.80% | 42.08% | 46.46% | 3.20 pp | -53 | 44 | -1.20 |
| BTC Market Hours | lstm | LSTM | 422 | 185 | 237 | 43.84% | 43.75% | 43.84% | 6.16 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| BTC Market Hours | rf | RandomForest | 422 | 182 | 240 | 43.13% | 42.92% | 43.13% | 6.87 pp | -58 | 42 | -1.38 |
| BTC Hourly | nn | NN | 827 | 373 | 454 | 45.10% | 42.50% | 44.58% | 4.90 pp | -81 | 44 | -1.84 |
| BTC Daily | lstm | LSTM | 650 | 288 | 362 | 44.31% | 42.08% | 43.75% | 5.69 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 476 | 197 | 279 | 41.39% | 42.50% | 41.39% | 8.61 pp | -82 | 42 | -1.95 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 37.92% | 39.81% | 10.19 pp | -86 | 42 | -2.05 |
| BTC Hourly | rf | RandomForest | 827 | 368 | 459 | 44.50% | 42.92% | 43.96% | 5.50 pp | -91 | 44 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 476 | 193 | 283 | 40.55% | 38.75% | 40.55% | 9.45 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 650 | 278 | 372 | 42.77% | 41.25% | 43.33% | 7.23 pp | -94 | 40 | -2.35 |
| BTC Hourly | lstm | LSTM | 827 | 359 | 468 | 43.41% | 40.83% | 43.54% | 6.59 pp | -109 | 44 | -2.48 |
| BTC Market Hours Daily | xgb | XGBoost | 476 | 185 | 291 | 38.87% | 35.42% | 38.87% | 11.13 pp | -106 | 42 | -2.52 |
| BTC Hourly | xgb | XGBoost | 827 | 349 | 478 | 42.20% | 38.75% | 42.29% | 7.80 pp | -129 | 44 | -2.93 |
| BTC Daily | xgb | XGBoost | 660 | 262 | 398 | 39.70% | 32.92% | 40.21% | 10.30 pp | -136 | 40 | -3.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 827 | 393 | 434 | 47.52% | 47.50% | 46.67% | 2.48 pp | -41 | 44 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 827 | 387 | 440 | 46.80% | 42.08% | 46.46% | 3.20 pp | -53 | 44 | -1.20 |
| BTC Hourly | nn | NN | 827 | 373 | 454 | 45.10% | 42.50% | 44.58% | 4.90 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 827 | 368 | 459 | 44.50% | 42.92% | 43.96% | 5.50 pp | -91 | 44 | -2.07 |
| BTC Hourly | lstm | LSTM | 827 | 359 | 468 | 43.41% | 40.83% | 43.54% | 6.59 pp | -109 | 44 | -2.48 |
| BTC Hourly | xgb | XGBoost | 827 | 349 | 478 | 42.20% | 38.75% | 42.29% | 7.80 pp | -129 | 44 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 650 | 317 | 333 | 48.77% | 46.25% | 50.00% | 1.23 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 650 | 315 | 335 | 48.46% | 45.83% | 49.58% | 1.54 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 650 | 304 | 346 | 46.77% | 42.08% | 48.96% | 3.23 pp | -42 | 40 | -1.05 |
| BTC Daily | lstm | LSTM | 650 | 288 | 362 | 44.31% | 42.08% | 43.75% | 5.69 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 650 | 278 | 372 | 42.77% | 41.25% | 43.33% | 7.23 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 660 | 262 | 398 | 39.70% | 32.92% | 40.21% | 10.30 pp | -136 | 40 | -3.40 |

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

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
