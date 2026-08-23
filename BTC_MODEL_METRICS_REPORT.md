# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T09:23:26.018010+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 08:00:00+00:00 | 1112 | 790 | 322 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 08:00:00+00:00 | 944 | 586 | 357 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 497 | 348 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 499 | 402 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 20:00:00+00:00 | 15 | 15 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 20:00:00+00:00 | 15 | 15 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 20:00:00+00:00 | 15 | 1 | 14 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 20:00:00+00:00 | 15 | 1 | 14 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 348 | 170 | 178 | 48.85% | 47.08% | 48.85% | 1.15 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 576 | 283 | 293 | 49.13% | 48.33% | 49.38% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 576 | 283 | 293 | 49.13% | 52.50% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 348 | 164 | 184 | 47.13% | 45.83% | 47.13% | 2.87 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 576 | 272 | 304 | 47.22% | 45.42% | 47.92% | 2.78 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 402 | 185 | 217 | 46.02% | 47.92% | 46.02% | 3.98 pp | -32 | 36 | -0.89 |
| BTC Market Hours | nn | NN | 348 | 158 | 190 | 45.40% | 47.50% | 45.40% | 4.60 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 402 | 183 | 219 | 45.52% | 45.42% | 45.52% | 4.48 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 402 | 183 | 219 | 45.52% | 46.25% | 45.52% | 4.48 pp | -36 | 36 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 348 | 149 | 199 | 42.82% | 42.50% | 42.82% | 7.18 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 348 | 147 | 201 | 42.24% | 42.50% | 42.24% | 7.76 pp | -54 | 36 | -1.50 |
| BTC Daily | lstm | LSTM | 576 | 259 | 317 | 44.97% | 45.42% | 44.58% | 5.03 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 348 | 145 | 203 | 41.67% | 42.08% | 41.67% | 8.33 pp | -58 | 36 | -1.61 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 402 | 165 | 237 | 41.04% | 39.58% | 41.04% | 8.96 pp | -72 | 36 | -2.00 |
| BTC Daily | rf | RandomForest | 576 | 249 | 327 | 43.23% | 44.17% | 43.75% | 6.77 pp | -78 | 37 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 402 | 162 | 240 | 40.30% | 39.58% | 40.30% | 9.70 pp | -78 | 36 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 402 | 161 | 241 | 40.05% | 37.92% | 40.05% | 9.95 pp | -80 | 36 | -2.22 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 586 | 236 | 350 | 40.27% | 35.00% | 40.62% | 9.73 pp | -114 | 37 | -3.08 |
| Consolidated Hourly | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 576 | 283 | 293 | 49.13% | 48.33% | 49.38% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 576 | 283 | 293 | 49.13% | 52.50% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 576 | 272 | 304 | 47.22% | 45.42% | 47.92% | 2.78 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 576 | 259 | 317 | 44.97% | 45.42% | 44.58% | 5.03 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 576 | 249 | 327 | 43.23% | 44.17% | 43.75% | 6.77 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 586 | 236 | 350 | 40.27% | 35.00% | 40.62% | 9.73 pp | -114 | 37 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 348 | 170 | 178 | 48.85% | 47.08% | 48.85% | 1.15 pp | -8 | 36 | -0.22 |
| BTC Market Hours | transformer | Transformer | 348 | 164 | 184 | 47.13% | 45.83% | 47.13% | 2.87 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 348 | 158 | 190 | 45.40% | 47.50% | 45.40% | 4.60 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 348 | 149 | 199 | 42.82% | 42.50% | 42.82% | 7.18 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 348 | 147 | 201 | 42.24% | 42.50% | 42.24% | 7.76 pp | -54 | 36 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 348 | 145 | 203 | 41.67% | 42.08% | 41.67% | 8.33 pp | -58 | 36 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 402 | 185 | 217 | 46.02% | 47.92% | 46.02% | 3.98 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 402 | 183 | 219 | 45.52% | 45.42% | 45.52% | 4.48 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 402 | 183 | 219 | 45.52% | 46.25% | 45.52% | 4.48 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 402 | 165 | 237 | 41.04% | 39.58% | 41.04% | 8.96 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 402 | 162 | 240 | 40.30% | 39.58% | 40.30% | 9.70 pp | -78 | 36 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 402 | 161 | 241 | 40.05% | 37.92% | 40.05% | 9.95 pp | -80 | 36 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 15 | 8 | 7 | 53.33% | 53.33% | 53.33% | 3.33 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
