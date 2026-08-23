# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T03:08:47.257801+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 02:00:00+00:00 | 1106 | 790 | 316 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 02:00:00+00:00 | 934 | 582 | 351 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 493 | 344 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 495 | 398 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 16:00:00+00:00 | 11 | 11 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 16:00:00+00:00 | 11 | 11 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 16:00:00+00:00 | 11 | 1 | 10 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 16:00:00+00:00 | 11 | 1 | 10 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 344 | 170 | 174 | 49.42% | 47.92% | 49.42% | 0.58 pp | -4 | 36 | -0.11 |
| BTC Daily | transformer | Transformer | 572 | 282 | 290 | 49.30% | 53.33% | 49.17% | 0.70 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 572 | 280 | 292 | 48.95% | 47.92% | 48.96% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Market Hours | transformer | Transformer | 344 | 163 | 181 | 47.38% | 46.67% | 47.38% | 2.62 pp | -18 | 36 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | nn | NN | 398 | 184 | 214 | 46.23% | 48.33% | 46.23% | 3.77 pp | -30 | 36 | -0.83 |
| BTC Daily | nn | NN | 572 | 270 | 302 | 47.20% | 45.83% | 47.92% | 2.80 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 344 | 156 | 188 | 45.35% | 47.50% | 45.35% | 4.65 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 398 | 182 | 216 | 45.73% | 46.25% | 45.73% | 4.27 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 398 | 181 | 217 | 45.48% | 45.42% | 45.48% | 4.52 pp | -36 | 36 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 344 | 149 | 195 | 43.31% | 43.75% | 43.31% | 6.69 pp | -46 | 36 | -1.28 |
| BTC Market Hours | rf | RandomForest | 344 | 146 | 198 | 42.44% | 42.92% | 42.44% | 7.56 pp | -52 | 36 | -1.44 |
| BTC Daily | lstm | LSTM | 572 | 258 | 314 | 45.10% | 46.25% | 44.58% | 4.90 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 344 | 143 | 201 | 41.57% | 42.08% | 41.57% | 8.43 pp | -58 | 36 | -1.61 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 398 | 164 | 234 | 41.21% | 39.58% | 41.21% | 8.79 pp | -70 | 36 | -1.94 |
| BTC Daily | rf | RandomForest | 572 | 248 | 324 | 43.36% | 45.42% | 43.75% | 6.64 pp | -76 | 37 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 398 | 162 | 236 | 40.70% | 40.00% | 40.70% | 9.30 pp | -74 | 36 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 398 | 158 | 240 | 39.70% | 37.50% | 39.70% | 10.30 pp | -82 | 36 | -2.28 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 582 | 236 | 346 | 40.55% | 36.25% | 40.83% | 9.45 pp | -110 | 37 | -2.97 |

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
| BTC Daily | transformer | Transformer | 572 | 282 | 290 | 49.30% | 53.33% | 49.17% | 0.70 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 572 | 280 | 292 | 48.95% | 47.92% | 48.96% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 572 | 270 | 302 | 47.20% | 45.83% | 47.92% | 2.80 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 572 | 258 | 314 | 45.10% | 46.25% | 44.58% | 4.90 pp | -56 | 37 | -1.51 |
| BTC Daily | rf | RandomForest | 572 | 248 | 324 | 43.36% | 45.42% | 43.75% | 6.64 pp | -76 | 37 | -2.05 |
| BTC Daily | xgb | XGBoost | 582 | 236 | 346 | 40.55% | 36.25% | 40.83% | 9.45 pp | -110 | 37 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 344 | 170 | 174 | 49.42% | 47.92% | 49.42% | 0.58 pp | -4 | 36 | -0.11 |
| BTC Market Hours | transformer | Transformer | 344 | 163 | 181 | 47.38% | 46.67% | 47.38% | 2.62 pp | -18 | 36 | -0.50 |
| BTC Market Hours | nn | NN | 344 | 156 | 188 | 45.35% | 47.50% | 45.35% | 4.65 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 344 | 149 | 195 | 43.31% | 43.75% | 43.31% | 6.69 pp | -46 | 36 | -1.28 |
| BTC Market Hours | rf | RandomForest | 344 | 146 | 198 | 42.44% | 42.92% | 42.44% | 7.56 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 344 | 143 | 201 | 41.57% | 42.08% | 41.57% | 8.43 pp | -58 | 36 | -1.61 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 398 | 184 | 214 | 46.23% | 48.33% | 46.23% | 3.77 pp | -30 | 36 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 398 | 182 | 216 | 45.73% | 46.25% | 45.73% | 4.27 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 398 | 181 | 217 | 45.48% | 45.42% | 45.48% | 4.52 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 398 | 164 | 234 | 41.21% | 39.58% | 41.21% | 8.79 pp | -70 | 36 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 398 | 162 | 236 | 40.70% | 40.00% | 40.70% | 9.30 pp | -74 | 36 | -2.06 |
| BTC Market Hours Daily | xgb | XGBoost | 398 | 158 | 240 | 39.70% | 37.50% | 39.70% | 10.30 pp | -82 | 36 | -2.28 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
