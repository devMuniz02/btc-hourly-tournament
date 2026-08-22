# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T09:40:53.969199+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 08:00:00+00:00 | 1088 | 790 | 298 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 08:00:00+00:00 | 906 | 572 | 333 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 470 | 334 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 472 | 388 | 82 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 334 | 165 | 169 | 49.40% | 48.33% | 49.40% | 0.60 pp | -4 | 35 | -0.11 |
| BTC Daily | transformer | Transformer | 562 | 276 | 286 | 49.11% | 52.08% | 49.17% | 0.89 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.92% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Market Hours | transformer | Transformer | 334 | 158 | 176 | 47.31% | 47.08% | 47.31% | 2.69 pp | -18 | 35 | -0.51 |
| BTC Market Hours Daily | nn | NN | 388 | 179 | 209 | 46.13% | 48.33% | 46.13% | 3.87 pp | -30 | 35 | -0.86 |
| BTC Daily | nn | NN | 562 | 265 | 297 | 47.15% | 44.58% | 47.29% | 2.85 pp | -32 | 36 | -0.89 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 388 | 176 | 212 | 45.36% | 45.42% | 45.36% | 4.64 pp | -36 | 35 | -1.03 |
| BTC Market Hours | nn | NN | 334 | 149 | 185 | 44.61% | 46.67% | 44.61% | 5.39 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 388 | 175 | 213 | 45.10% | 45.42% | 45.10% | 4.90 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 334 | 145 | 189 | 43.41% | 44.17% | 43.41% | 6.59 pp | -44 | 35 | -1.26 |
| BTC Daily | lstm | LSTM | 562 | 254 | 308 | 45.20% | 45.42% | 44.79% | 4.80 pp | -54 | 36 | -1.50 |
| BTC Market Hours | rf | RandomForest | 334 | 139 | 195 | 41.62% | 42.08% | 41.62% | 8.38 pp | -56 | 35 | -1.60 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 334 | 135 | 199 | 40.42% | 40.00% | 40.42% | 9.58 pp | -64 | 35 | -1.83 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 562 | 245 | 317 | 43.59% | 45.00% | 44.38% | 6.41 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 388 | 158 | 230 | 40.72% | 37.92% | 40.72% | 9.28 pp | -72 | 35 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 388 | 156 | 232 | 40.21% | 38.33% | 40.21% | 9.79 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 388 | 153 | 235 | 39.43% | 37.50% | 39.43% | 10.57 pp | -82 | 35 | -2.34 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 572 | 232 | 340 | 40.56% | 36.25% | 40.42% | 9.44 pp | -108 | 36 | -3.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

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
| BTC Daily | transformer | Transformer | 562 | 276 | 286 | 49.11% | 52.08% | 49.17% | 0.89 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 562 | 274 | 288 | 48.75% | 47.92% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Daily | nn | NN | 562 | 265 | 297 | 47.15% | 44.58% | 47.29% | 2.85 pp | -32 | 36 | -0.89 |
| BTC Daily | lstm | LSTM | 562 | 254 | 308 | 45.20% | 45.42% | 44.79% | 4.80 pp | -54 | 36 | -1.50 |
| BTC Daily | rf | RandomForest | 562 | 245 | 317 | 43.59% | 45.00% | 44.38% | 6.41 pp | -72 | 36 | -2.00 |
| BTC Daily | xgb | XGBoost | 572 | 232 | 340 | 40.56% | 36.25% | 40.42% | 9.44 pp | -108 | 36 | -3.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 334 | 165 | 169 | 49.40% | 48.33% | 49.40% | 0.60 pp | -4 | 35 | -0.11 |
| BTC Market Hours | transformer | Transformer | 334 | 158 | 176 | 47.31% | 47.08% | 47.31% | 2.69 pp | -18 | 35 | -0.51 |
| BTC Market Hours | nn | NN | 334 | 149 | 185 | 44.61% | 46.67% | 44.61% | 5.39 pp | -36 | 35 | -1.03 |
| BTC Market Hours | lstm | LSTM | 334 | 145 | 189 | 43.41% | 44.17% | 43.41% | 6.59 pp | -44 | 35 | -1.26 |
| BTC Market Hours | rf | RandomForest | 334 | 139 | 195 | 41.62% | 42.08% | 41.62% | 8.38 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 334 | 135 | 199 | 40.42% | 40.00% | 40.42% | 9.58 pp | -64 | 35 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 388 | 179 | 209 | 46.13% | 48.33% | 46.13% | 3.87 pp | -30 | 35 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 388 | 176 | 212 | 45.36% | 45.42% | 45.36% | 4.64 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 388 | 175 | 213 | 45.10% | 45.42% | 45.10% | 4.90 pp | -38 | 35 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 388 | 158 | 230 | 40.72% | 37.92% | 40.72% | 9.28 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 388 | 156 | 232 | 40.21% | 38.33% | 40.21% | 9.79 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 388 | 153 | 235 | 39.43% | 37.50% | 39.43% | 10.57 pp | -82 | 35 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

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
