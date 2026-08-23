# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T14:10:08.742332+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 13:00:00+00:00 | 1117 | 790 | 327 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 13:00:00+00:00 | 952 | 589 | 362 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 13:00:00+00:00 | 502 | 351 | 150 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 13:00:00+00:00 | 504 | 405 | 97 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 1 | 16 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 23:00:00+00:00 | 17 | 1 | 16 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 351 | 172 | 179 | 49.00% | 46.67% | 49.00% | 1.00 pp | -7 | 36 | -0.19 |
| BTC Daily | transformer | Transformer | 579 | 285 | 294 | 49.22% | 52.08% | 49.17% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 579 | 284 | 295 | 49.05% | 48.75% | 49.38% | 0.95 pp | -11 | 37 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 351 | 165 | 186 | 47.01% | 45.83% | 47.01% | 2.99 pp | -21 | 36 | -0.58 |
| BTC Market Hours | nn | NN | 351 | 160 | 191 | 45.58% | 47.50% | 45.58% | 4.42 pp | -31 | 36 | -0.86 |
| BTC Daily | nn | NN | 579 | 273 | 306 | 47.15% | 45.83% | 47.92% | 2.85 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 405 | 186 | 219 | 45.93% | 47.50% | 45.93% | 4.07 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 405 | 185 | 220 | 45.68% | 45.83% | 45.68% | 4.32 pp | -35 | 36 | -0.97 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 405 | 184 | 221 | 45.43% | 46.67% | 45.43% | 4.57 pp | -37 | 36 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 351 | 150 | 201 | 42.74% | 42.50% | 42.74% | 7.26 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 351 | 149 | 202 | 42.45% | 42.08% | 42.45% | 7.55 pp | -53 | 36 | -1.47 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 351 | 146 | 205 | 41.60% | 42.08% | 41.60% | 8.40 pp | -59 | 36 | -1.64 |
| BTC Daily | lstm | LSTM | 579 | 259 | 320 | 44.73% | 45.42% | 44.38% | 5.27 pp | -61 | 37 | -1.65 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 405 | 166 | 239 | 40.99% | 39.58% | 40.99% | 9.01 pp | -73 | 36 | -2.03 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 579 | 249 | 330 | 43.01% | 44.17% | 43.54% | 6.99 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 405 | 163 | 242 | 40.25% | 38.75% | 40.25% | 9.75 pp | -79 | 36 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 405 | 162 | 243 | 40.00% | 37.50% | 40.00% | 10.00 pp | -81 | 36 | -2.25 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 589 | 236 | 353 | 40.07% | 35.00% | 40.42% | 9.93 pp | -117 | 37 | -3.16 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

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
| BTC Daily | transformer | Transformer | 579 | 285 | 294 | 49.22% | 52.08% | 49.17% | 0.78 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 579 | 284 | 295 | 49.05% | 48.75% | 49.38% | 0.95 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 579 | 273 | 306 | 47.15% | 45.83% | 47.92% | 2.85 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 579 | 259 | 320 | 44.73% | 45.42% | 44.38% | 5.27 pp | -61 | 37 | -1.65 |
| BTC Daily | rf | RandomForest | 579 | 249 | 330 | 43.01% | 44.17% | 43.54% | 6.99 pp | -81 | 37 | -2.19 |
| BTC Daily | xgb | XGBoost | 589 | 236 | 353 | 40.07% | 35.00% | 40.42% | 9.93 pp | -117 | 37 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 351 | 172 | 179 | 49.00% | 46.67% | 49.00% | 1.00 pp | -7 | 36 | -0.19 |
| BTC Market Hours | transformer | Transformer | 351 | 165 | 186 | 47.01% | 45.83% | 47.01% | 2.99 pp | -21 | 36 | -0.58 |
| BTC Market Hours | nn | NN | 351 | 160 | 191 | 45.58% | 47.50% | 45.58% | 4.42 pp | -31 | 36 | -0.86 |
| BTC Market Hours | lstm | LSTM | 351 | 150 | 201 | 42.74% | 42.50% | 42.74% | 7.26 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 351 | 149 | 202 | 42.45% | 42.08% | 42.45% | 7.55 pp | -53 | 36 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 351 | 146 | 205 | 41.60% | 42.08% | 41.60% | 8.40 pp | -59 | 36 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 405 | 186 | 219 | 45.93% | 47.50% | 45.93% | 4.07 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 405 | 185 | 220 | 45.68% | 45.83% | 45.68% | 4.32 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 405 | 184 | 221 | 45.43% | 46.67% | 45.43% | 4.57 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 405 | 166 | 239 | 40.99% | 39.58% | 40.99% | 9.01 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 405 | 163 | 242 | 40.25% | 38.75% | 40.25% | 9.75 pp | -79 | 36 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 405 | 162 | 243 | 40.00% | 37.50% | 40.00% | 10.00 pp | -81 | 36 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
