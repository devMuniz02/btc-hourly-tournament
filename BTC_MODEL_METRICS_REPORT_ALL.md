# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T07:06:09.354310+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 06:00:00+00:00 | 902 | 570 | 331 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 468 | 332 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 470 | 386 | 82 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 17:00:00+00:00 | 6 | 6 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 17:00:00+00:00 | 6 | 6 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 17:00:00+00:00 | 6 | 1 | 5 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 17:00:00+00:00 | 6 | 1 | 5 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Hourly | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 332 | 164 | 168 | 49.40% | 48.33% | 49.40% | 0.60 pp | -4 | 35 | -0.11 |
| BTC Daily | transformer | Transformer | 560 | 276 | 284 | 49.29% | 52.50% | 49.38% | 0.71 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 560 | 273 | 287 | 48.75% | 48.33% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Market Hours | transformer | Transformer | 332 | 156 | 176 | 46.99% | 46.67% | 46.99% | 3.01 pp | -20 | 35 | -0.57 |
| BTC Market Hours Daily | nn | NN | 386 | 177 | 209 | 45.85% | 48.33% | 45.85% | 4.15 pp | -32 | 35 | -0.91 |
| BTC Daily | nn | NN | 560 | 263 | 297 | 46.96% | 44.58% | 47.08% | 3.04 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 386 | 175 | 211 | 45.34% | 46.25% | 45.34% | 4.66 pp | -36 | 35 | -1.03 |
| BTC Market Hours | nn | NN | 332 | 148 | 184 | 44.58% | 46.67% | 44.58% | 5.42 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 386 | 174 | 212 | 45.08% | 45.42% | 45.08% | 4.92 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 332 | 145 | 187 | 43.67% | 44.58% | 43.67% | 6.33 pp | -42 | 35 | -1.20 |
| BTC Daily | lstm | LSTM | 560 | 254 | 306 | 45.36% | 45.42% | 45.00% | 4.64 pp | -52 | 36 | -1.44 |
| BTC Market Hours | rf | RandomForest | 332 | 138 | 194 | 41.57% | 42.08% | 41.57% | 8.43 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 332 | 135 | 197 | 40.66% | 40.42% | 40.66% | 9.34 pp | -62 | 35 | -1.77 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 560 | 245 | 315 | 43.75% | 45.42% | 44.38% | 6.25 pp | -70 | 36 | -1.94 |
| BTC Market Hours Daily | rf | RandomForest | 386 | 158 | 228 | 40.93% | 38.75% | 40.93% | 9.07 pp | -70 | 35 | -2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Hourly | nn | NN | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 386 | 156 | 230 | 40.41% | 39.17% | 40.41% | 9.59 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 386 | 153 | 233 | 39.64% | 37.92% | 39.64% | 10.36 pp | -80 | 35 | -2.29 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 570 | 232 | 338 | 40.70% | 36.67% | 40.83% | 9.30 pp | -106 | 36 | -2.94 |

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
| BTC Daily | transformer | Transformer | 560 | 276 | 284 | 49.29% | 52.50% | 49.38% | 0.71 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 560 | 273 | 287 | 48.75% | 48.33% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Daily | nn | NN | 560 | 263 | 297 | 46.96% | 44.58% | 47.08% | 3.04 pp | -34 | 36 | -0.94 |
| BTC Daily | lstm | LSTM | 560 | 254 | 306 | 45.36% | 45.42% | 45.00% | 4.64 pp | -52 | 36 | -1.44 |
| BTC Daily | rf | RandomForest | 560 | 245 | 315 | 43.75% | 45.42% | 44.38% | 6.25 pp | -70 | 36 | -1.94 |
| BTC Daily | xgb | XGBoost | 570 | 232 | 338 | 40.70% | 36.67% | 40.83% | 9.30 pp | -106 | 36 | -2.94 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 332 | 164 | 168 | 49.40% | 48.33% | 49.40% | 0.60 pp | -4 | 35 | -0.11 |
| BTC Market Hours | transformer | Transformer | 332 | 156 | 176 | 46.99% | 46.67% | 46.99% | 3.01 pp | -20 | 35 | -0.57 |
| BTC Market Hours | nn | NN | 332 | 148 | 184 | 44.58% | 46.67% | 44.58% | 5.42 pp | -36 | 35 | -1.03 |
| BTC Market Hours | lstm | LSTM | 332 | 145 | 187 | 43.67% | 44.58% | 43.67% | 6.33 pp | -42 | 35 | -1.20 |
| BTC Market Hours | rf | RandomForest | 332 | 138 | 194 | 41.57% | 42.08% | 41.57% | 8.43 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 332 | 135 | 197 | 40.66% | 40.42% | 40.66% | 9.34 pp | -62 | 35 | -1.77 |

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
| Consolidated Hourly | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Hourly | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Hourly | nn | NN | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
