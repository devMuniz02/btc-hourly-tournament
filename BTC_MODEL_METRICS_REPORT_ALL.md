# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T11:26:49.715983+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 10:00:00+00:00 | 1115 | 791 | 324 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 10:00:00+00:00 | 947 | 587 | 359 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 498 | 349 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 500 | 403 | 95 | 2 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 577 | 284 | 293 | 49.22% | 48.75% | 49.58% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 349 | 170 | 179 | 48.71% | 46.67% | 48.71% | 1.29 pp | -9 | 36 | -0.25 |
| BTC Daily | transformer | Transformer | 577 | 283 | 294 | 49.05% | 52.08% | 48.96% | 0.95 pp | -11 | 37 | -0.30 |
| Consolidated Hourly | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 349 | 164 | 185 | 46.99% | 45.83% | 46.99% | 3.01 pp | -21 | 36 | -0.58 |
| BTC Daily | nn | NN | 577 | 273 | 304 | 47.31% | 45.83% | 48.12% | 2.69 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | nn | NN | 403 | 185 | 218 | 45.91% | 47.50% | 45.91% | 4.09 pp | -33 | 36 | -0.92 |
| BTC Market Hours | nn | NN | 349 | 158 | 191 | 45.27% | 47.50% | 45.27% | 4.73 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 403 | 183 | 220 | 45.41% | 45.00% | 45.41% | 4.59 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 403 | 183 | 220 | 45.41% | 46.25% | 45.41% | 4.59 pp | -37 | 36 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 349 | 150 | 199 | 42.98% | 42.92% | 42.98% | 7.02 pp | -49 | 36 | -1.36 |
| BTC Market Hours | rf | RandomForest | 349 | 147 | 202 | 42.12% | 42.08% | 42.12% | 7.88 pp | -55 | 36 | -1.53 |
| BTC Daily | lstm | LSTM | 577 | 259 | 318 | 44.89% | 45.42% | 44.58% | 5.11 pp | -59 | 37 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 349 | 145 | 204 | 41.55% | 42.08% | 41.55% | 8.45 pp | -59 | 36 | -1.64 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 403 | 165 | 238 | 40.94% | 39.58% | 40.94% | 9.06 pp | -73 | 36 | -2.03 |
| BTC Daily | rf | RandomForest | 577 | 249 | 328 | 43.15% | 44.17% | 43.75% | 6.85 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 403 | 162 | 241 | 40.20% | 39.17% | 40.20% | 9.80 pp | -79 | 36 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 403 | 161 | 242 | 39.95% | 37.50% | 39.95% | 10.05 pp | -81 | 36 | -2.25 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| BTC Daily | xgb | XGBoost | 587 | 236 | 351 | 40.20% | 35.00% | 40.42% | 9.80 pp | -115 | 37 | -3.11 |
| Consolidated Hourly | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 577 | 284 | 293 | 49.22% | 48.75% | 49.58% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Daily | transformer | Transformer | 577 | 283 | 294 | 49.05% | 52.08% | 48.96% | 0.95 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 577 | 273 | 304 | 47.31% | 45.83% | 48.12% | 2.69 pp | -31 | 37 | -0.84 |
| BTC Daily | lstm | LSTM | 577 | 259 | 318 | 44.89% | 45.42% | 44.58% | 5.11 pp | -59 | 37 | -1.59 |
| BTC Daily | rf | RandomForest | 577 | 249 | 328 | 43.15% | 44.17% | 43.75% | 6.85 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 587 | 236 | 351 | 40.20% | 35.00% | 40.42% | 9.80 pp | -115 | 37 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 349 | 170 | 179 | 48.71% | 46.67% | 48.71% | 1.29 pp | -9 | 36 | -0.25 |
| BTC Market Hours | transformer | Transformer | 349 | 164 | 185 | 46.99% | 45.83% | 46.99% | 3.01 pp | -21 | 36 | -0.58 |
| BTC Market Hours | nn | NN | 349 | 158 | 191 | 45.27% | 47.50% | 45.27% | 4.73 pp | -33 | 36 | -0.92 |
| BTC Market Hours | lstm | LSTM | 349 | 150 | 199 | 42.98% | 42.92% | 42.98% | 7.02 pp | -49 | 36 | -1.36 |
| BTC Market Hours | rf | RandomForest | 349 | 147 | 202 | 42.12% | 42.08% | 42.12% | 7.88 pp | -55 | 36 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 349 | 145 | 204 | 41.55% | 42.08% | 41.55% | 8.45 pp | -59 | 36 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 403 | 185 | 218 | 45.91% | 47.50% | 45.91% | 4.09 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 403 | 183 | 220 | 45.41% | 45.00% | 45.41% | 4.59 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 403 | 183 | 220 | 45.41% | 46.25% | 45.41% | 4.59 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 403 | 165 | 238 | 40.94% | 39.58% | 40.94% | 9.06 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 403 | 162 | 241 | 40.20% | 39.17% | 40.20% | 9.80 pp | -79 | 36 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 403 | 161 | 242 | 39.95% | 37.50% | 39.95% | 10.05 pp | -81 | 36 | -2.25 |

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
