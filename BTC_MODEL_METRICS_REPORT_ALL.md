# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T16:01:15.364832+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1148 | 860 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1024 | 659 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 15:00:00+00:00 | 631 | 421 | 209 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 15:00:00+00:00 | 632 | 474 | 156 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 74 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 0 | 74 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 03:00:00+00:00 | 74 | 0 | 74 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 421 | 208 | 213 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 649 | 316 | 333 | 48.69% | 46.25% | 49.79% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 649 | 314 | 335 | 48.38% | 45.42% | 49.38% | 1.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 421 | 198 | 223 | 47.03% | 50.42% | 47.03% | 2.97 pp | -25 | 42 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 474 | 219 | 255 | 46.20% | 46.67% | 46.20% | 3.80 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 421 | 192 | 229 | 45.61% | 41.25% | 45.61% | 4.39 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 826 | 393 | 433 | 47.58% | 47.50% | 46.67% | 2.42 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 649 | 304 | 345 | 46.84% | 42.50% | 48.96% | 3.16 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | nn | NN | 474 | 215 | 259 | 45.36% | 44.58% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 474 | 215 | 259 | 45.36% | 45.00% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 826 | 386 | 440 | 46.73% | 42.08% | 46.25% | 3.27 pp | -54 | 44 | -1.23 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 421 | 184 | 237 | 43.71% | 43.33% | 43.71% | 6.29 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 421 | 182 | 239 | 43.23% | 42.92% | 43.23% | 6.77 pp | -57 | 42 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 826 | 372 | 454 | 45.04% | 42.08% | 44.58% | 4.96 pp | -82 | 44 | -1.86 |
| BTC Daily | lstm | LSTM | 649 | 287 | 362 | 44.22% | 42.08% | 43.75% | 5.78 pp | -75 | 40 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 474 | 196 | 278 | 41.35% | 42.50% | 41.35% | 8.65 pp | -82 | 42 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 421 | 168 | 253 | 39.90% | 37.92% | 39.90% | 10.10 pp | -85 | 42 | -2.02 |
| BTC Hourly | rf | RandomForest | 826 | 368 | 458 | 44.55% | 43.33% | 43.96% | 5.45 pp | -90 | 44 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 474 | 192 | 282 | 40.51% | 38.75% | 40.51% | 9.49 pp | -90 | 42 | -2.14 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 649 | 277 | 372 | 42.68% | 41.25% | 43.33% | 7.32 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 826 | 359 | 467 | 43.46% | 41.25% | 43.75% | 6.54 pp | -108 | 44 | -2.45 |
| BTC Market Hours Daily | xgb | XGBoost | 474 | 184 | 290 | 38.82% | 35.42% | 38.82% | 11.18 pp | -106 | 42 | -2.52 |
| BTC Hourly | xgb | XGBoost | 826 | 348 | 478 | 42.13% | 38.75% | 42.29% | 7.87 pp | -130 | 44 | -2.95 |
| BTC Daily | xgb | XGBoost | 659 | 261 | 398 | 39.61% | 32.50% | 40.00% | 10.39 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 826 | 393 | 433 | 47.58% | 47.50% | 46.67% | 2.42 pp | -40 | 44 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 826 | 386 | 440 | 46.73% | 42.08% | 46.25% | 3.27 pp | -54 | 44 | -1.23 |
| BTC Hourly | nn | NN | 826 | 372 | 454 | 45.04% | 42.08% | 44.58% | 4.96 pp | -82 | 44 | -1.86 |
| BTC Hourly | rf | RandomForest | 826 | 368 | 458 | 44.55% | 43.33% | 43.96% | 5.45 pp | -90 | 44 | -2.05 |
| BTC Hourly | lstm | LSTM | 826 | 359 | 467 | 43.46% | 41.25% | 43.75% | 6.54 pp | -108 | 44 | -2.45 |
| BTC Hourly | xgb | XGBoost | 826 | 348 | 478 | 42.13% | 38.75% | 42.29% | 7.87 pp | -130 | 44 | -2.95 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 649 | 316 | 333 | 48.69% | 46.25% | 49.79% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 649 | 314 | 335 | 48.38% | 45.42% | 49.38% | 1.62 pp | -21 | 40 | -0.53 |
| BTC Daily | nn | NN | 649 | 304 | 345 | 46.84% | 42.50% | 48.96% | 3.16 pp | -41 | 40 | -1.02 |
| BTC Daily | lstm | LSTM | 649 | 287 | 362 | 44.22% | 42.08% | 43.75% | 5.78 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 649 | 277 | 372 | 42.68% | 41.25% | 43.33% | 7.32 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 659 | 261 | 398 | 39.61% | 32.50% | 40.00% | 10.39 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 421 | 208 | 213 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Market Hours | nn | NN | 421 | 198 | 223 | 47.03% | 50.42% | 47.03% | 2.97 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 421 | 192 | 229 | 45.61% | 41.25% | 45.61% | 4.39 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 421 | 184 | 237 | 43.71% | 43.33% | 43.71% | 6.29 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 421 | 182 | 239 | 43.23% | 42.92% | 43.23% | 6.77 pp | -57 | 42 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 421 | 168 | 253 | 39.90% | 37.92% | 39.90% | 10.10 pp | -85 | 42 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 474 | 219 | 255 | 46.20% | 46.67% | 46.20% | 3.80 pp | -36 | 42 | -0.86 |
| BTC Market Hours Daily | nn | NN | 474 | 215 | 259 | 45.36% | 44.58% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 474 | 215 | 259 | 45.36% | 45.00% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 474 | 196 | 278 | 41.35% | 42.50% | 41.35% | 8.65 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 474 | 192 | 282 | 40.51% | 38.75% | 40.51% | 9.49 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 474 | 184 | 290 | 38.82% | 35.42% | 38.82% | 11.18 pp | -106 | 42 | -2.52 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

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
