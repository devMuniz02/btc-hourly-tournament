# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T07:16:43.788859+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 06:00:00+00:00 | 1110 | 790 | 320 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 06:00:00+00:00 | 941 | 585 | 355 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 496 | 347 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 498 | 401 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 347 | 170 | 177 | 48.99% | 47.50% | 48.99% | 1.01 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 575 | 283 | 292 | 49.22% | 48.33% | 49.38% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Daily | transformer | Transformer | 575 | 282 | 293 | 49.04% | 52.08% | 48.96% | 0.96 pp | -11 | 37 | -0.30 |
| BTC Market Hours | transformer | Transformer | 347 | 163 | 184 | 46.97% | 45.83% | 46.97% | 3.03 pp | -21 | 36 | -0.58 |
| BTC Daily | nn | NN | 575 | 272 | 303 | 47.30% | 45.83% | 48.12% | 2.70 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | nn | NN | 401 | 184 | 217 | 45.89% | 47.92% | 45.89% | 4.11 pp | -33 | 36 | -0.92 |
| BTC Market Hours | nn | NN | 347 | 157 | 190 | 45.24% | 47.08% | 45.24% | 4.76 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 401 | 183 | 218 | 45.64% | 46.25% | 45.64% | 4.36 pp | -35 | 36 | -0.97 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 401 | 182 | 219 | 45.39% | 45.42% | 45.39% | 4.61 pp | -37 | 36 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 347 | 149 | 198 | 42.94% | 42.92% | 42.94% | 7.06 pp | -49 | 36 | -1.36 |
| BTC Market Hours | rf | RandomForest | 347 | 147 | 200 | 42.36% | 42.92% | 42.36% | 7.64 pp | -53 | 36 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 347 | 145 | 202 | 41.79% | 42.08% | 41.79% | 8.21 pp | -57 | 36 | -1.58 |
| BTC Daily | lstm | LSTM | 575 | 258 | 317 | 44.87% | 45.00% | 44.38% | 5.13 pp | -59 | 37 | -1.59 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 401 | 164 | 237 | 40.90% | 39.58% | 40.90% | 9.10 pp | -73 | 36 | -2.03 |
| BTC Daily | rf | RandomForest | 575 | 248 | 327 | 43.13% | 44.17% | 43.54% | 6.87 pp | -79 | 37 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 401 | 162 | 239 | 40.40% | 40.00% | 40.40% | 9.60 pp | -77 | 36 | -2.14 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 38.33% | 40.15% | 9.85 pp | -79 | 36 | -2.19 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 585 | 236 | 349 | 40.34% | 35.42% | 40.62% | 9.66 pp | -113 | 37 | -3.05 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 575 | 283 | 292 | 49.22% | 48.33% | 49.38% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Daily | transformer | Transformer | 575 | 282 | 293 | 49.04% | 52.08% | 48.96% | 0.96 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 575 | 272 | 303 | 47.30% | 45.83% | 48.12% | 2.70 pp | -31 | 37 | -0.84 |
| BTC Daily | lstm | LSTM | 575 | 258 | 317 | 44.87% | 45.00% | 44.38% | 5.13 pp | -59 | 37 | -1.59 |
| BTC Daily | rf | RandomForest | 575 | 248 | 327 | 43.13% | 44.17% | 43.54% | 6.87 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 585 | 236 | 349 | 40.34% | 35.42% | 40.62% | 9.66 pp | -113 | 37 | -3.05 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 347 | 170 | 177 | 48.99% | 47.50% | 48.99% | 1.01 pp | -7 | 36 | -0.19 |
| BTC Market Hours | transformer | Transformer | 347 | 163 | 184 | 46.97% | 45.83% | 46.97% | 3.03 pp | -21 | 36 | -0.58 |
| BTC Market Hours | nn | NN | 347 | 157 | 190 | 45.24% | 47.08% | 45.24% | 4.76 pp | -33 | 36 | -0.92 |
| BTC Market Hours | lstm | LSTM | 347 | 149 | 198 | 42.94% | 42.92% | 42.94% | 7.06 pp | -49 | 36 | -1.36 |
| BTC Market Hours | rf | RandomForest | 347 | 147 | 200 | 42.36% | 42.92% | 42.36% | 7.64 pp | -53 | 36 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 347 | 145 | 202 | 41.79% | 42.08% | 41.79% | 8.21 pp | -57 | 36 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 401 | 184 | 217 | 45.89% | 47.92% | 45.89% | 4.11 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 401 | 183 | 218 | 45.64% | 46.25% | 45.64% | 4.36 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 401 | 182 | 219 | 45.39% | 45.42% | 45.39% | 4.61 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 401 | 164 | 237 | 40.90% | 39.58% | 40.90% | 9.10 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 401 | 162 | 239 | 40.40% | 40.00% | 40.40% | 9.60 pp | -77 | 36 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 38.33% | 40.15% | 9.85 pp | -79 | 36 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

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
