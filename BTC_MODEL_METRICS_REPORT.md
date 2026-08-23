# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T12:04:02.998473+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 11:00:00+00:00 | 1115 | 790 | 325 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 11:00:00+00:00 | 949 | 588 | 360 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 499 | 350 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 501 | 404 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 10:00:00+00:00 | 15 | 15 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 10:00:00+00:00 | 15 | 15 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 10:00:00+00:00 | 15 | 0 | 15 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 10:00:00+00:00 | 15 | 0 | 15 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 350 | 171 | 179 | 48.86% | 46.67% | 48.86% | 1.14 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 578 | 284 | 294 | 49.13% | 48.75% | 49.38% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 578 | 284 | 294 | 49.13% | 52.08% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 350 | 165 | 185 | 47.14% | 45.83% | 47.14% | 2.86 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 578 | 273 | 305 | 47.23% | 45.83% | 47.92% | 2.77 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 350 | 159 | 191 | 45.43% | 47.50% | 45.43% | 4.57 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | nn | NN | 404 | 185 | 219 | 45.79% | 47.50% | 45.79% | 4.21 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 404 | 184 | 220 | 45.54% | 45.42% | 45.54% | 4.46 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 404 | 183 | 221 | 45.30% | 46.25% | 45.30% | 4.70 pp | -38 | 36 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 350 | 150 | 200 | 42.86% | 42.92% | 42.86% | 7.14 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 350 | 148 | 202 | 42.29% | 42.08% | 42.29% | 7.71 pp | -54 | 36 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 350 | 146 | 204 | 41.71% | 42.50% | 41.71% | 8.29 pp | -58 | 36 | -1.61 |
| BTC Daily | lstm | LSTM | 578 | 259 | 319 | 44.81% | 45.42% | 44.38% | 5.19 pp | -60 | 37 | -1.62 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 404 | 165 | 239 | 40.84% | 39.17% | 40.84% | 9.16 pp | -74 | 36 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 578 | 249 | 329 | 43.08% | 44.17% | 43.54% | 6.92 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 404 | 163 | 241 | 40.35% | 39.17% | 40.35% | 9.65 pp | -78 | 36 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 404 | 161 | 243 | 39.85% | 37.08% | 39.85% | 10.15 pp | -82 | 36 | -2.28 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 588 | 236 | 352 | 40.14% | 35.00% | 40.42% | 9.86 pp | -116 | 37 | -3.14 |
| Consolidated Hourly | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 578 | 284 | 294 | 49.13% | 48.75% | 49.38% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 578 | 284 | 294 | 49.13% | 52.08% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 578 | 273 | 305 | 47.23% | 45.83% | 47.92% | 2.77 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 578 | 259 | 319 | 44.81% | 45.42% | 44.38% | 5.19 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 578 | 249 | 329 | 43.08% | 44.17% | 43.54% | 6.92 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 588 | 236 | 352 | 40.14% | 35.00% | 40.42% | 9.86 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 350 | 171 | 179 | 48.86% | 46.67% | 48.86% | 1.14 pp | -8 | 36 | -0.22 |
| BTC Market Hours | transformer | Transformer | 350 | 165 | 185 | 47.14% | 45.83% | 47.14% | 2.86 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 350 | 159 | 191 | 45.43% | 47.50% | 45.43% | 4.57 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 350 | 150 | 200 | 42.86% | 42.92% | 42.86% | 7.14 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 350 | 148 | 202 | 42.29% | 42.08% | 42.29% | 7.71 pp | -54 | 36 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 350 | 146 | 204 | 41.71% | 42.50% | 41.71% | 8.29 pp | -58 | 36 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 404 | 185 | 219 | 45.79% | 47.50% | 45.79% | 4.21 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 404 | 184 | 220 | 45.54% | 45.42% | 45.54% | 4.46 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 404 | 183 | 221 | 45.30% | 46.25% | 45.30% | 4.70 pp | -38 | 36 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 404 | 165 | 239 | 40.84% | 39.17% | 40.84% | 9.16 pp | -74 | 36 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 404 | 163 | 241 | 40.35% | 39.17% | 40.35% | 9.65 pp | -78 | 36 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 404 | 161 | 243 | 39.85% | 37.08% | 39.85% | 10.15 pp | -82 | 36 | -2.28 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

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
