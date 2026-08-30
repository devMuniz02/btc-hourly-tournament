# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T17:02:11.141419+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 632 | 421 | 210 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 634 | 475 | 157 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 0 | 75 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 0 | 75 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 421 | 208 | 213 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 649 | 316 | 333 | 48.69% | 46.25% | 49.79% | 1.31 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 649 | 314 | 335 | 48.38% | 45.42% | 49.38% | 1.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 421 | 198 | 223 | 47.03% | 50.42% | 47.03% | 2.97 pp | -25 | 42 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 475 | 219 | 256 | 46.11% | 46.67% | 46.11% | 3.89 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 421 | 192 | 229 | 45.61% | 41.25% | 45.61% | 4.39 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 826 | 393 | 433 | 47.58% | 47.50% | 46.67% | 2.42 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 649 | 304 | 345 | 46.84% | 42.50% | 48.96% | 3.16 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | nn | NN | 475 | 215 | 260 | 45.26% | 44.17% | 45.26% | 4.74 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 475 | 215 | 260 | 45.26% | 44.58% | 45.26% | 4.74 pp | -45 | 42 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 826 | 387 | 439 | 46.85% | 42.50% | 46.46% | 3.15 pp | -52 | 44 | -1.18 |
| BTC Market Hours | lstm | LSTM | 421 | 184 | 237 | 43.71% | 43.33% | 43.71% | 6.29 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 421 | 182 | 239 | 43.23% | 42.92% | 43.23% | 6.77 pp | -57 | 42 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| BTC Hourly | nn | NN | 826 | 373 | 453 | 45.16% | 42.50% | 44.79% | 4.84 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 649 | 287 | 362 | 44.22% | 42.08% | 43.75% | 5.78 pp | -75 | 40 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 475 | 196 | 279 | 41.26% | 42.08% | 41.26% | 8.74 pp | -83 | 42 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 421 | 168 | 253 | 39.90% | 37.92% | 39.90% | 10.10 pp | -85 | 42 | -2.02 |
| BTC Hourly | rf | RandomForest | 826 | 368 | 458 | 44.55% | 43.33% | 43.96% | 5.45 pp | -90 | 44 | -2.05 |
| Consolidated Hourly | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 475 | 192 | 283 | 40.42% | 38.33% | 40.42% | 9.58 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 649 | 277 | 372 | 42.68% | 41.25% | 43.33% | 7.32 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 826 | 359 | 467 | 43.46% | 41.25% | 43.75% | 6.54 pp | -108 | 44 | -2.45 |
| BTC Market Hours Daily | xgb | XGBoost | 475 | 184 | 291 | 38.74% | 35.00% | 38.74% | 11.26 pp | -107 | 42 | -2.55 |
| BTC Hourly | xgb | XGBoost | 826 | 349 | 477 | 42.25% | 39.17% | 42.50% | 7.75 pp | -128 | 44 | -2.91 |
| BTC Daily | xgb | XGBoost | 659 | 261 | 398 | 39.61% | 32.50% | 40.00% | 10.39 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 826 | 393 | 433 | 47.58% | 47.50% | 46.67% | 2.42 pp | -40 | 44 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 826 | 387 | 439 | 46.85% | 42.50% | 46.46% | 3.15 pp | -52 | 44 | -1.18 |
| BTC Hourly | nn | NN | 826 | 373 | 453 | 45.16% | 42.50% | 44.79% | 4.84 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 826 | 368 | 458 | 44.55% | 43.33% | 43.96% | 5.45 pp | -90 | 44 | -2.05 |
| BTC Hourly | lstm | LSTM | 826 | 359 | 467 | 43.46% | 41.25% | 43.75% | 6.54 pp | -108 | 44 | -2.45 |
| BTC Hourly | xgb | XGBoost | 826 | 349 | 477 | 42.25% | 39.17% | 42.50% | 7.75 pp | -128 | 44 | -2.91 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 475 | 219 | 256 | 46.11% | 46.67% | 46.11% | 3.89 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | nn | NN | 475 | 215 | 260 | 45.26% | 44.17% | 45.26% | 4.74 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 475 | 215 | 260 | 45.26% | 44.58% | 45.26% | 4.74 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 475 | 196 | 279 | 41.26% | 42.08% | 41.26% | 8.74 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 475 | 192 | 283 | 40.42% | 38.33% | 40.42% | 9.58 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 475 | 184 | 291 | 38.74% | 35.00% | 38.74% | 11.26 pp | -107 | 42 | -2.55 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |

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
