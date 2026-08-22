# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T07:46:52.267445+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 06:00:00+00:00 | 1086 | 790 | 296 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 06:00:00+00:00 | 903 | 571 | 331 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 469 | 333 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 470 | 386 | 82 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 333 | 164 | 169 | 49.25% | 47.92% | 49.25% | 0.75 pp | -5 | 35 | -0.14 |
| BTC Daily | transformer | Transformer | 561 | 276 | 285 | 49.20% | 52.08% | 49.17% | 0.80 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 561 | 273 | 288 | 48.66% | 47.92% | 48.54% | 1.34 pp | -15 | 36 | -0.42 |
| BTC Market Hours | transformer | Transformer | 333 | 157 | 176 | 47.15% | 46.67% | 47.15% | 2.85 pp | -19 | 35 | -0.54 |
| BTC Market Hours Daily | nn | NN | 386 | 177 | 209 | 45.85% | 48.33% | 45.85% | 4.15 pp | -32 | 35 | -0.91 |
| BTC Daily | nn | NN | 561 | 264 | 297 | 47.06% | 44.58% | 47.29% | 2.94 pp | -33 | 36 | -0.92 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 386 | 175 | 211 | 45.34% | 46.25% | 45.34% | 4.66 pp | -36 | 35 | -1.03 |
| BTC Market Hours | nn | NN | 333 | 148 | 185 | 44.44% | 46.25% | 44.44% | 5.56 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 386 | 174 | 212 | 45.08% | 45.42% | 45.08% | 4.92 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 333 | 145 | 188 | 43.54% | 44.17% | 43.54% | 6.46 pp | -43 | 35 | -1.23 |
| BTC Daily | lstm | LSTM | 561 | 254 | 307 | 45.28% | 45.42% | 45.00% | 4.72 pp | -53 | 36 | -1.47 |
| BTC Market Hours | rf | RandomForest | 333 | 138 | 195 | 41.44% | 42.08% | 41.44% | 8.56 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 333 | 135 | 198 | 40.54% | 40.42% | 40.54% | 9.46 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 561 | 245 | 316 | 43.67% | 45.00% | 44.38% | 6.33 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | rf | RandomForest | 386 | 158 | 228 | 40.93% | 38.75% | 40.93% | 9.07 pp | -70 | 35 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 386 | 156 | 230 | 40.41% | 39.17% | 40.41% | 9.59 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 386 | 153 | 233 | 39.64% | 37.92% | 39.64% | 10.36 pp | -80 | 35 | -2.29 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 571 | 232 | 339 | 40.63% | 36.25% | 40.62% | 9.37 pp | -107 | 36 | -2.97 |
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
| BTC Daily | transformer | Transformer | 561 | 276 | 285 | 49.20% | 52.08% | 49.17% | 0.80 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 561 | 273 | 288 | 48.66% | 47.92% | 48.54% | 1.34 pp | -15 | 36 | -0.42 |
| BTC Daily | nn | NN | 561 | 264 | 297 | 47.06% | 44.58% | 47.29% | 2.94 pp | -33 | 36 | -0.92 |
| BTC Daily | lstm | LSTM | 561 | 254 | 307 | 45.28% | 45.42% | 45.00% | 4.72 pp | -53 | 36 | -1.47 |
| BTC Daily | rf | RandomForest | 561 | 245 | 316 | 43.67% | 45.00% | 44.38% | 6.33 pp | -71 | 36 | -1.97 |
| BTC Daily | xgb | XGBoost | 571 | 232 | 339 | 40.63% | 36.25% | 40.62% | 9.37 pp | -107 | 36 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 333 | 164 | 169 | 49.25% | 47.92% | 49.25% | 0.75 pp | -5 | 35 | -0.14 |
| BTC Market Hours | transformer | Transformer | 333 | 157 | 176 | 47.15% | 46.67% | 47.15% | 2.85 pp | -19 | 35 | -0.54 |
| BTC Market Hours | nn | NN | 333 | 148 | 185 | 44.44% | 46.25% | 44.44% | 5.56 pp | -37 | 35 | -1.06 |
| BTC Market Hours | lstm | LSTM | 333 | 145 | 188 | 43.54% | 44.17% | 43.54% | 6.46 pp | -43 | 35 | -1.23 |
| BTC Market Hours | rf | RandomForest | 333 | 138 | 195 | 41.44% | 42.08% | 41.44% | 8.56 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 333 | 135 | 198 | 40.54% | 40.42% | 40.54% | 9.46 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 386 | 177 | 209 | 45.85% | 48.33% | 45.85% | 4.15 pp | -32 | 35 | -0.91 |
| BTC Market Hours Daily | transformer | Transformer | 386 | 175 | 211 | 45.34% | 46.25% | 45.34% | 4.66 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 386 | 174 | 212 | 45.08% | 45.42% | 45.08% | 4.92 pp | -38 | 35 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 386 | 158 | 228 | 40.93% | 38.75% | 40.93% | 9.07 pp | -70 | 35 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 386 | 156 | 230 | 40.41% | 39.17% | 40.41% | 9.59 pp | -74 | 35 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 386 | 153 | 233 | 39.64% | 37.92% | 39.64% | 10.36 pp | -80 | 35 | -2.29 |

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
