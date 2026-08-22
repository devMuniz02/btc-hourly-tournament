# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T17:44:13.543692+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 16:00:00+00:00 | 1096 | 790 | 306 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 16:00:00+00:00 | 918 | 576 | 341 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 16:00:00+00:00 | 479 | 338 | 140 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 16:00:00+00:00 | 481 | 392 | 87 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 338 | 168 | 170 | 49.70% | 48.33% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Daily | transformer | Transformer | 566 | 278 | 288 | 49.12% | 52.50% | 48.96% | 0.88 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 566 | 276 | 290 | 48.76% | 47.50% | 48.96% | 1.24 pp | -14 | 36 | -0.39 |
| BTC Market Hours | transformer | Transformer | 338 | 159 | 179 | 47.04% | 46.67% | 47.04% | 2.96 pp | -20 | 35 | -0.57 |
| BTC Market Hours Daily | nn | NN | 392 | 182 | 210 | 46.43% | 48.33% | 46.43% | 3.57 pp | -28 | 35 | -0.80 |
| BTC Daily | nn | NN | 566 | 268 | 298 | 47.35% | 45.83% | 47.92% | 2.65 pp | -30 | 36 | -0.83 |
| BTC Market Hours | nn | NN | 338 | 152 | 186 | 44.97% | 47.08% | 44.97% | 5.03 pp | -34 | 35 | -0.97 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 392 | 178 | 214 | 45.41% | 45.00% | 45.41% | 4.59 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 392 | 178 | 214 | 45.41% | 45.42% | 45.41% | 4.59 pp | -36 | 35 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 338 | 148 | 190 | 43.79% | 44.17% | 43.79% | 6.21 pp | -42 | 35 | -1.20 |
| BTC Daily | lstm | LSTM | 566 | 256 | 310 | 45.23% | 45.83% | 45.00% | 4.77 pp | -54 | 36 | -1.50 |
| BTC Market Hours | rf | RandomForest | 338 | 141 | 197 | 41.72% | 41.67% | 41.72% | 8.28 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 338 | 138 | 200 | 40.83% | 40.42% | 40.83% | 9.17 pp | -62 | 35 | -1.77 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 566 | 246 | 320 | 43.46% | 45.42% | 44.17% | 6.54 pp | -74 | 36 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 392 | 159 | 233 | 40.56% | 37.50% | 40.56% | 9.44 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 392 | 158 | 234 | 40.31% | 38.33% | 40.31% | 9.69 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 392 | 153 | 239 | 39.03% | 35.83% | 39.03% | 10.97 pp | -86 | 35 | -2.46 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 576 | 233 | 343 | 40.45% | 35.83% | 40.42% | 9.55 pp | -110 | 36 | -3.06 |

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
| BTC Daily | transformer | Transformer | 566 | 278 | 288 | 49.12% | 52.50% | 48.96% | 0.88 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 566 | 276 | 290 | 48.76% | 47.50% | 48.96% | 1.24 pp | -14 | 36 | -0.39 |
| BTC Daily | nn | NN | 566 | 268 | 298 | 47.35% | 45.83% | 47.92% | 2.65 pp | -30 | 36 | -0.83 |
| BTC Daily | lstm | LSTM | 566 | 256 | 310 | 45.23% | 45.83% | 45.00% | 4.77 pp | -54 | 36 | -1.50 |
| BTC Daily | rf | RandomForest | 566 | 246 | 320 | 43.46% | 45.42% | 44.17% | 6.54 pp | -74 | 36 | -2.06 |
| BTC Daily | xgb | XGBoost | 576 | 233 | 343 | 40.45% | 35.83% | 40.42% | 9.55 pp | -110 | 36 | -3.06 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 338 | 168 | 170 | 49.70% | 48.33% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Market Hours | transformer | Transformer | 338 | 159 | 179 | 47.04% | 46.67% | 47.04% | 2.96 pp | -20 | 35 | -0.57 |
| BTC Market Hours | nn | NN | 338 | 152 | 186 | 44.97% | 47.08% | 44.97% | 5.03 pp | -34 | 35 | -0.97 |
| BTC Market Hours | lstm | LSTM | 338 | 148 | 190 | 43.79% | 44.17% | 43.79% | 6.21 pp | -42 | 35 | -1.20 |
| BTC Market Hours | rf | RandomForest | 338 | 141 | 197 | 41.72% | 41.67% | 41.72% | 8.28 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 338 | 138 | 200 | 40.83% | 40.42% | 40.83% | 9.17 pp | -62 | 35 | -1.77 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 392 | 182 | 210 | 46.43% | 48.33% | 46.43% | 3.57 pp | -28 | 35 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 392 | 178 | 214 | 45.41% | 45.00% | 45.41% | 4.59 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 392 | 178 | 214 | 45.41% | 45.42% | 45.41% | 4.59 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 392 | 159 | 233 | 40.56% | 37.50% | 40.56% | 9.44 pp | -74 | 35 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 392 | 158 | 234 | 40.31% | 38.33% | 40.31% | 9.69 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 392 | 153 | 239 | 39.03% | 35.83% | 39.03% | 10.97 pp | -86 | 35 | -2.46 |

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
