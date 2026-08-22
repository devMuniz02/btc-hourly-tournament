# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T18:09:31.599665+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 17:00:00+00:00 | 1097 | 790 | 307 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 17:00:00+00:00 | 920 | 577 | 342 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 17:00:00+00:00 | 481 | 339 | 141 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 17:00:00+00:00 | 483 | 393 | 88 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 339 | 169 | 170 | 49.85% | 48.33% | 49.85% | 0.15 pp | -1 | 35 | -0.03 |
| BTC Daily | transformer | Transformer | 567 | 279 | 288 | 49.21% | 52.92% | 48.96% | 0.79 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 567 | 277 | 290 | 48.85% | 47.50% | 49.17% | 1.15 pp | -13 | 36 | -0.36 |
| BTC Market Hours | transformer | Transformer | 339 | 159 | 180 | 46.90% | 46.67% | 46.90% | 3.10 pp | -21 | 35 | -0.60 |
| BTC Daily | nn | NN | 567 | 269 | 298 | 47.44% | 46.25% | 48.12% | 2.56 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | nn | NN | 393 | 182 | 211 | 46.31% | 47.92% | 46.31% | 3.69 pp | -29 | 35 | -0.83 |
| BTC Market Hours | nn | NN | 339 | 153 | 186 | 45.13% | 47.50% | 45.13% | 4.87 pp | -33 | 35 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 393 | 179 | 214 | 45.55% | 45.00% | 45.55% | 4.45 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 393 | 178 | 215 | 45.29% | 45.42% | 45.29% | 4.71 pp | -37 | 35 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 339 | 149 | 190 | 43.95% | 44.58% | 43.95% | 6.05 pp | -41 | 35 | -1.17 |
| BTC Daily | lstm | LSTM | 567 | 256 | 311 | 45.15% | 45.83% | 45.00% | 4.85 pp | -55 | 36 | -1.53 |
| BTC Market Hours | rf | RandomForest | 339 | 142 | 197 | 41.89% | 41.67% | 41.89% | 8.11 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 339 | 139 | 200 | 41.00% | 40.83% | 41.00% | 9.00 pp | -61 | 35 | -1.74 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 567 | 247 | 320 | 43.56% | 45.83% | 44.17% | 6.44 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | rf | RandomForest | 393 | 160 | 233 | 40.71% | 37.92% | 40.71% | 9.29 pp | -73 | 35 | -2.09 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 393 | 159 | 234 | 40.46% | 38.75% | 40.46% | 9.54 pp | -75 | 35 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 393 | 154 | 239 | 39.19% | 36.25% | 39.19% | 10.81 pp | -85 | 35 | -2.43 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 577 | 234 | 343 | 40.55% | 35.83% | 40.62% | 9.45 pp | -109 | 36 | -3.03 |

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
| BTC Daily | transformer | Transformer | 567 | 279 | 288 | 49.21% | 52.92% | 48.96% | 0.79 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 567 | 277 | 290 | 48.85% | 47.50% | 49.17% | 1.15 pp | -13 | 36 | -0.36 |
| BTC Daily | nn | NN | 567 | 269 | 298 | 47.44% | 46.25% | 48.12% | 2.56 pp | -29 | 36 | -0.81 |
| BTC Daily | lstm | LSTM | 567 | 256 | 311 | 45.15% | 45.83% | 45.00% | 4.85 pp | -55 | 36 | -1.53 |
| BTC Daily | rf | RandomForest | 567 | 247 | 320 | 43.56% | 45.83% | 44.17% | 6.44 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 577 | 234 | 343 | 40.55% | 35.83% | 40.62% | 9.45 pp | -109 | 36 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 339 | 169 | 170 | 49.85% | 48.33% | 49.85% | 0.15 pp | -1 | 35 | -0.03 |
| BTC Market Hours | transformer | Transformer | 339 | 159 | 180 | 46.90% | 46.67% | 46.90% | 3.10 pp | -21 | 35 | -0.60 |
| BTC Market Hours | nn | NN | 339 | 153 | 186 | 45.13% | 47.50% | 45.13% | 4.87 pp | -33 | 35 | -0.94 |
| BTC Market Hours | lstm | LSTM | 339 | 149 | 190 | 43.95% | 44.58% | 43.95% | 6.05 pp | -41 | 35 | -1.17 |
| BTC Market Hours | rf | RandomForest | 339 | 142 | 197 | 41.89% | 41.67% | 41.89% | 8.11 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 339 | 139 | 200 | 41.00% | 40.83% | 41.00% | 9.00 pp | -61 | 35 | -1.74 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 393 | 182 | 211 | 46.31% | 47.92% | 46.31% | 3.69 pp | -29 | 35 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 393 | 179 | 214 | 45.55% | 45.00% | 45.55% | 4.45 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 393 | 178 | 215 | 45.29% | 45.42% | 45.29% | 4.71 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 393 | 160 | 233 | 40.71% | 37.92% | 40.71% | 9.29 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 393 | 159 | 234 | 40.46% | 38.75% | 40.46% | 9.54 pp | -75 | 35 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 393 | 154 | 239 | 39.19% | 36.25% | 39.19% | 10.81 pp | -85 | 35 | -2.43 |

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
