# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T01:15:10.406826+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 1080 | 790 | 290 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 893 | 567 | 325 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 465 | 329 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 466 | 382 | 82 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 2 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 2 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 0 | 2 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 0 | 2 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 329 | 163 | 166 | 49.54% | 48.75% | 49.54% | 0.46 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 557 | 274 | 283 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 48.75% | 48.75% | 1.17 pp | -13 | 36 | -0.36 |
| BTC Market Hours | transformer | Transformer | 329 | 155 | 174 | 47.11% | 47.08% | 47.11% | 2.89 pp | -19 | 35 | -0.54 |
| BTC Market Hours Daily | nn | NN | 382 | 176 | 206 | 46.07% | 48.33% | 46.07% | 3.93 pp | -30 | 35 | -0.86 |
| BTC Daily | nn | NN | 557 | 262 | 295 | 47.04% | 45.00% | 47.08% | 2.96 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 382 | 173 | 209 | 45.29% | 46.25% | 45.29% | 4.71 pp | -36 | 35 | -1.03 |
| BTC Market Hours | nn | NN | 329 | 146 | 183 | 44.38% | 46.67% | 44.38% | 5.62 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 382 | 172 | 210 | 45.03% | 45.42% | 45.03% | 4.97 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 329 | 145 | 184 | 44.07% | 45.42% | 44.07% | 5.93 pp | -39 | 35 | -1.11 |
| BTC Daily | lstm | LSTM | 557 | 253 | 304 | 45.42% | 45.83% | 45.21% | 4.58 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 329 | 136 | 193 | 41.34% | 42.08% | 41.34% | 8.66 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 329 | 133 | 196 | 40.43% | 40.42% | 40.43% | 9.57 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 557 | 242 | 315 | 43.45% | 45.00% | 43.96% | 6.55 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 382 | 155 | 227 | 40.58% | 38.75% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 382 | 155 | 227 | 40.58% | 37.92% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 382 | 151 | 231 | 39.53% | 37.50% | 39.53% | 10.47 pp | -80 | 35 | -2.29 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 567 | 229 | 338 | 40.39% | 36.25% | 40.62% | 9.61 pp | -109 | 36 | -3.03 |

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
| BTC Daily | transformer | Transformer | 557 | 274 | 283 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 48.75% | 48.75% | 1.17 pp | -13 | 36 | -0.36 |
| BTC Daily | nn | NN | 557 | 262 | 295 | 47.04% | 45.00% | 47.08% | 2.96 pp | -33 | 36 | -0.92 |
| BTC Daily | lstm | LSTM | 557 | 253 | 304 | 45.42% | 45.83% | 45.21% | 4.58 pp | -51 | 36 | -1.42 |
| BTC Daily | rf | RandomForest | 557 | 242 | 315 | 43.45% | 45.00% | 43.96% | 6.55 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 567 | 229 | 338 | 40.39% | 36.25% | 40.62% | 9.61 pp | -109 | 36 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 329 | 163 | 166 | 49.54% | 48.75% | 49.54% | 0.46 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 329 | 155 | 174 | 47.11% | 47.08% | 47.11% | 2.89 pp | -19 | 35 | -0.54 |
| BTC Market Hours | nn | NN | 329 | 146 | 183 | 44.38% | 46.67% | 44.38% | 5.62 pp | -37 | 35 | -1.06 |
| BTC Market Hours | lstm | LSTM | 329 | 145 | 184 | 44.07% | 45.42% | 44.07% | 5.93 pp | -39 | 35 | -1.11 |
| BTC Market Hours | rf | RandomForest | 329 | 136 | 193 | 41.34% | 42.08% | 41.34% | 8.66 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 329 | 133 | 196 | 40.43% | 40.42% | 40.43% | 9.57 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 382 | 176 | 206 | 46.07% | 48.33% | 46.07% | 3.93 pp | -30 | 35 | -0.86 |
| BTC Market Hours Daily | transformer | Transformer | 382 | 173 | 209 | 45.29% | 46.25% | 45.29% | 4.71 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 382 | 172 | 210 | 45.03% | 45.42% | 45.03% | 4.97 pp | -38 | 35 | -1.09 |
| BTC Market Hours Daily | lstm | LSTM | 382 | 155 | 227 | 40.58% | 38.75% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 382 | 155 | 227 | 40.58% | 37.92% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | xgb | XGBoost | 382 | 151 | 231 | 39.53% | 37.50% | 39.53% | 10.47 pp | -80 | 35 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |

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
