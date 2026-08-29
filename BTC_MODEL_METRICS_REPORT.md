# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T18:15:32.959634+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1131 | 843 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1007 | 642 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 603 | 404 | 198 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 604 | 457 | 145 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 60 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 60 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 0 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 0 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 404 | 201 | 203 | 49.75% | 49.17% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Daily | transformer | Transformer | 632 | 310 | 322 | 49.05% | 47.92% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| Consolidated Hourly | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 632 | 308 | 324 | 48.73% | 46.25% | 50.00% | 1.27 pp | -16 | 39 | -0.41 |
| BTC Market Hours | nn | NN | 404 | 192 | 212 | 47.52% | 51.25% | 47.52% | 2.48 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 404 | 187 | 217 | 46.29% | 42.92% | 46.29% | 3.71 pp | -30 | 40 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 457 | 212 | 245 | 46.39% | 46.67% | 46.39% | 3.61 pp | -33 | 40 | -0.82 |
| BTC Daily | nn | NN | 632 | 297 | 335 | 46.99% | 43.33% | 48.96% | 3.01 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 457 | 209 | 248 | 45.73% | 46.25% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 457 | 209 | 248 | 45.73% | 47.08% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 809 | 381 | 428 | 47.10% | 44.17% | 46.88% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 809 | 381 | 428 | 47.10% | 44.58% | 46.04% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 404 | 179 | 225 | 44.31% | 46.25% | 44.31% | 5.69 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 404 | 174 | 230 | 43.07% | 42.50% | 43.07% | 6.93 pp | -56 | 40 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 632 | 280 | 352 | 44.30% | 42.50% | 43.75% | 5.70 pp | -72 | 39 | -1.85 |
| BTC Hourly | nn | NN | 809 | 363 | 446 | 44.87% | 40.42% | 44.58% | 5.13 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 404 | 163 | 241 | 40.35% | 38.75% | 40.35% | 9.65 pp | -78 | 40 | -1.95 |
| BTC Hourly | rf | RandomForest | 809 | 361 | 448 | 44.62% | 44.17% | 44.38% | 5.38 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 457 | 188 | 269 | 41.14% | 40.83% | 41.14% | 8.86 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 457 | 187 | 270 | 40.92% | 40.00% | 40.92% | 9.08 pp | -83 | 40 | -2.08 |
| BTC Hourly | lstm | LSTM | 809 | 354 | 455 | 43.76% | 42.08% | 44.58% | 6.24 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 632 | 270 | 362 | 42.72% | 42.50% | 43.54% | 7.28 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 457 | 180 | 277 | 39.39% | 37.08% | 39.39% | 10.61 pp | -97 | 40 | -2.42 |
| BTC Hourly | xgb | XGBoost | 809 | 343 | 466 | 42.40% | 40.00% | 42.92% | 7.60 pp | -123 | 44 | -2.80 |
| Consolidated Hourly | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| BTC Daily | xgb | XGBoost | 642 | 251 | 391 | 39.10% | 30.83% | 39.17% | 10.90 pp | -140 | 39 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 809 | 381 | 428 | 47.10% | 44.17% | 46.88% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 809 | 381 | 428 | 47.10% | 44.58% | 46.04% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 809 | 363 | 446 | 44.87% | 40.42% | 44.58% | 5.13 pp | -83 | 44 | -1.89 |
| BTC Hourly | rf | RandomForest | 809 | 361 | 448 | 44.62% | 44.17% | 44.38% | 5.38 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 809 | 354 | 455 | 43.76% | 42.08% | 44.58% | 6.24 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 809 | 343 | 466 | 42.40% | 40.00% | 42.92% | 7.60 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 632 | 310 | 322 | 49.05% | 47.92% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 632 | 308 | 324 | 48.73% | 46.25% | 50.00% | 1.27 pp | -16 | 39 | -0.41 |
| BTC Daily | nn | NN | 632 | 297 | 335 | 46.99% | 43.33% | 48.96% | 3.01 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 632 | 280 | 352 | 44.30% | 42.50% | 43.75% | 5.70 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 632 | 270 | 362 | 42.72% | 42.50% | 43.54% | 7.28 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 642 | 251 | 391 | 39.10% | 30.83% | 39.17% | 10.90 pp | -140 | 39 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 404 | 201 | 203 | 49.75% | 49.17% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Market Hours | nn | NN | 404 | 192 | 212 | 47.52% | 51.25% | 47.52% | 2.48 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 404 | 187 | 217 | 46.29% | 42.92% | 46.29% | 3.71 pp | -30 | 40 | -0.75 |
| BTC Market Hours | lstm | LSTM | 404 | 179 | 225 | 44.31% | 46.25% | 44.31% | 5.69 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 404 | 174 | 230 | 43.07% | 42.50% | 43.07% | 6.93 pp | -56 | 40 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 404 | 163 | 241 | 40.35% | 38.75% | 40.35% | 9.65 pp | -78 | 40 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 457 | 212 | 245 | 46.39% | 46.67% | 46.39% | 3.61 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | nn | NN | 457 | 209 | 248 | 45.73% | 46.25% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 457 | 209 | 248 | 45.73% | 47.08% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 457 | 188 | 269 | 41.14% | 40.83% | 41.14% | 8.86 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 457 | 187 | 270 | 40.92% | 40.00% | 40.92% | 9.08 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 457 | 180 | 277 | 39.39% | 37.08% | 39.39% | 10.61 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |

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
