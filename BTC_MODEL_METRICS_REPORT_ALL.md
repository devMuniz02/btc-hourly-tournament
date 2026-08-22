# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T12:41:37.005208+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 11:00:00+00:00 | 1091 | 790 | 301 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 11:00:00+00:00 | 910 | 573 | 336 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 471 | 335 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 473 | 389 | 82 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 335 | 166 | 169 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 563 | 277 | 286 | 49.20% | 52.08% | 49.17% | 0.80 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 563 | 275 | 288 | 48.85% | 47.92% | 48.75% | 1.15 pp | -13 | 36 | -0.36 |
| BTC Market Hours | transformer | Transformer | 335 | 158 | 177 | 47.16% | 46.67% | 47.16% | 2.84 pp | -19 | 35 | -0.54 |
| BTC Market Hours Daily | nn | NN | 389 | 180 | 209 | 46.27% | 48.33% | 46.27% | 3.73 pp | -29 | 35 | -0.83 |
| BTC Daily | nn | NN | 563 | 266 | 297 | 47.25% | 45.00% | 47.50% | 2.75 pp | -31 | 36 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 389 | 177 | 212 | 45.50% | 45.42% | 45.50% | 4.50 pp | -35 | 35 | -1.00 |
| BTC Market Hours | nn | NN | 335 | 150 | 185 | 44.78% | 46.67% | 44.78% | 5.22 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 389 | 176 | 213 | 45.24% | 45.42% | 45.24% | 4.76 pp | -37 | 35 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 335 | 146 | 189 | 43.58% | 44.17% | 43.58% | 6.42 pp | -43 | 35 | -1.23 |
| BTC Daily | lstm | LSTM | 563 | 254 | 309 | 45.12% | 45.42% | 44.79% | 4.88 pp | -55 | 36 | -1.53 |
| BTC Market Hours | rf | RandomForest | 335 | 140 | 195 | 41.79% | 42.08% | 41.79% | 8.21 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 335 | 136 | 199 | 40.60% | 40.00% | 40.60% | 9.40 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 563 | 244 | 319 | 43.34% | 44.58% | 44.17% | 6.66 pp | -75 | 36 | -2.08 |
| BTC Market Hours Daily | rf | RandomForest | 389 | 158 | 231 | 40.62% | 37.50% | 40.62% | 9.38 pp | -73 | 35 | -2.09 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 389 | 156 | 233 | 40.10% | 38.33% | 40.10% | 9.90 pp | -77 | 35 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 389 | 153 | 236 | 39.33% | 37.08% | 39.33% | 10.67 pp | -83 | 35 | -2.37 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 573 | 231 | 342 | 40.31% | 35.83% | 40.00% | 9.69 pp | -111 | 36 | -3.08 |

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
| BTC Daily | transformer | Transformer | 563 | 277 | 286 | 49.20% | 52.08% | 49.17% | 0.80 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 563 | 275 | 288 | 48.85% | 47.92% | 48.75% | 1.15 pp | -13 | 36 | -0.36 |
| BTC Daily | nn | NN | 563 | 266 | 297 | 47.25% | 45.00% | 47.50% | 2.75 pp | -31 | 36 | -0.86 |
| BTC Daily | lstm | LSTM | 563 | 254 | 309 | 45.12% | 45.42% | 44.79% | 4.88 pp | -55 | 36 | -1.53 |
| BTC Daily | rf | RandomForest | 563 | 244 | 319 | 43.34% | 44.58% | 44.17% | 6.66 pp | -75 | 36 | -2.08 |
| BTC Daily | xgb | XGBoost | 573 | 231 | 342 | 40.31% | 35.83% | 40.00% | 9.69 pp | -111 | 36 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 335 | 166 | 169 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 335 | 158 | 177 | 47.16% | 46.67% | 47.16% | 2.84 pp | -19 | 35 | -0.54 |
| BTC Market Hours | nn | NN | 335 | 150 | 185 | 44.78% | 46.67% | 44.78% | 5.22 pp | -35 | 35 | -1.00 |
| BTC Market Hours | lstm | LSTM | 335 | 146 | 189 | 43.58% | 44.17% | 43.58% | 6.42 pp | -43 | 35 | -1.23 |
| BTC Market Hours | rf | RandomForest | 335 | 140 | 195 | 41.79% | 42.08% | 41.79% | 8.21 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 335 | 136 | 199 | 40.60% | 40.00% | 40.60% | 9.40 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 389 | 180 | 209 | 46.27% | 48.33% | 46.27% | 3.73 pp | -29 | 35 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 389 | 177 | 212 | 45.50% | 45.42% | 45.50% | 4.50 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 389 | 176 | 213 | 45.24% | 45.42% | 45.24% | 4.76 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 389 | 158 | 231 | 40.62% | 37.50% | 40.62% | 9.38 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 389 | 156 | 233 | 40.10% | 38.33% | 40.10% | 9.90 pp | -77 | 35 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 389 | 153 | 236 | 39.33% | 37.08% | 39.33% | 10.67 pp | -83 | 35 | -2.37 |

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
