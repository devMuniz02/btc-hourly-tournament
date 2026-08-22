# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T14:37:05.513480+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 1093 | 790 | 303 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 914 | 575 | 338 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 475 | 337 | 137 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 476 | 390 | 84 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 337 | 167 | 170 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 565 | 279 | 286 | 49.38% | 52.92% | 49.38% | 0.62 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 565 | 277 | 288 | 49.03% | 48.33% | 49.17% | 0.97 pp | -11 | 36 | -0.31 |
| BTC Market Hours | transformer | Transformer | 337 | 158 | 179 | 46.88% | 46.25% | 46.88% | 3.12 pp | -21 | 35 | -0.60 |
| BTC Market Hours Daily | nn | NN | 390 | 181 | 209 | 46.41% | 48.75% | 46.41% | 3.59 pp | -28 | 35 | -0.80 |
| BTC Daily | nn | NN | 565 | 268 | 297 | 47.43% | 45.83% | 47.92% | 2.57 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 390 | 178 | 212 | 45.64% | 45.42% | 45.64% | 4.36 pp | -34 | 35 | -0.97 |
| BTC Market Hours | nn | NN | 337 | 151 | 186 | 44.81% | 46.67% | 44.81% | 5.19 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 390 | 177 | 213 | 45.38% | 45.83% | 45.38% | 4.62 pp | -36 | 35 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 337 | 147 | 190 | 43.62% | 44.17% | 43.62% | 6.38 pp | -43 | 35 | -1.23 |
| BTC Daily | lstm | LSTM | 565 | 255 | 310 | 45.13% | 45.83% | 44.79% | 4.87 pp | -55 | 36 | -1.53 |
| BTC Market Hours | rf | RandomForest | 337 | 140 | 197 | 41.54% | 41.67% | 41.54% | 8.46 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 337 | 137 | 200 | 40.65% | 40.00% | 40.65% | 9.35 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 565 | 246 | 319 | 43.54% | 45.42% | 44.38% | 6.46 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | rf | RandomForest | 390 | 158 | 232 | 40.51% | 37.50% | 40.51% | 9.49 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 390 | 157 | 233 | 40.26% | 38.33% | 40.26% | 9.74 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 390 | 153 | 237 | 39.23% | 36.67% | 39.23% | 10.77 pp | -84 | 35 | -2.40 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 575 | 233 | 342 | 40.52% | 36.25% | 40.42% | 9.48 pp | -109 | 36 | -3.03 |

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
| BTC Daily | transformer | Transformer | 565 | 279 | 286 | 49.38% | 52.92% | 49.38% | 0.62 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 565 | 277 | 288 | 49.03% | 48.33% | 49.17% | 0.97 pp | -11 | 36 | -0.31 |
| BTC Daily | nn | NN | 565 | 268 | 297 | 47.43% | 45.83% | 47.92% | 2.57 pp | -29 | 36 | -0.81 |
| BTC Daily | lstm | LSTM | 565 | 255 | 310 | 45.13% | 45.83% | 44.79% | 4.87 pp | -55 | 36 | -1.53 |
| BTC Daily | rf | RandomForest | 565 | 246 | 319 | 43.54% | 45.42% | 44.38% | 6.46 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 575 | 233 | 342 | 40.52% | 36.25% | 40.42% | 9.48 pp | -109 | 36 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 337 | 167 | 170 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 337 | 158 | 179 | 46.88% | 46.25% | 46.88% | 3.12 pp | -21 | 35 | -0.60 |
| BTC Market Hours | nn | NN | 337 | 151 | 186 | 44.81% | 46.67% | 44.81% | 5.19 pp | -35 | 35 | -1.00 |
| BTC Market Hours | lstm | LSTM | 337 | 147 | 190 | 43.62% | 44.17% | 43.62% | 6.38 pp | -43 | 35 | -1.23 |
| BTC Market Hours | rf | RandomForest | 337 | 140 | 197 | 41.54% | 41.67% | 41.54% | 8.46 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 337 | 137 | 200 | 40.65% | 40.00% | 40.65% | 9.35 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 390 | 181 | 209 | 46.41% | 48.75% | 46.41% | 3.59 pp | -28 | 35 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 390 | 178 | 212 | 45.64% | 45.42% | 45.64% | 4.36 pp | -34 | 35 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 390 | 177 | 213 | 45.38% | 45.83% | 45.38% | 4.62 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 390 | 158 | 232 | 40.51% | 37.50% | 40.51% | 9.49 pp | -74 | 35 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 390 | 157 | 233 | 40.26% | 38.33% | 40.26% | 9.74 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 390 | 153 | 237 | 39.23% | 36.67% | 39.23% | 10.77 pp | -84 | 35 | -2.40 |

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
