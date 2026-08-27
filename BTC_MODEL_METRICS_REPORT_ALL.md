# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T21:47:42.590189+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 808 | 311 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 972 | 607 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 20:00:00+00:00 | 545 | 369 | 175 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 20:00:00+00:00 | 547 | 423 | 122 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T04:00:00+00:00 | 31 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T04:00:00+00:00 | 31 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T04:00:00+00:00 | 31 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-21T04:00:00+00:00 | 32 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 597 | 295 | 302 | 49.41% | 50.83% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 369 | 181 | 188 | 49.05% | 48.75% | 49.05% | 0.95 pp | -7 | 38 | -0.18 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 597 | 291 | 306 | 48.74% | 47.08% | 49.17% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Market Hours | transformer | Transformer | 369 | 174 | 195 | 47.15% | 45.42% | 47.15% | 2.85 pp | -21 | 38 | -0.55 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 369 | 170 | 199 | 46.07% | 48.33% | 46.07% | 3.93 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 597 | 281 | 316 | 47.07% | 45.42% | 48.12% | 2.93 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 423 | 193 | 230 | 45.63% | 45.00% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 423 | 193 | 230 | 45.63% | 46.25% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 423 | 193 | 230 | 45.63% | 47.50% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 774 | 364 | 410 | 47.03% | 43.33% | 46.88% | 2.97 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 774 | 362 | 412 | 46.77% | 42.50% | 45.62% | 3.23 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 369 | 157 | 212 | 42.55% | 42.92% | 42.55% | 7.45 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 369 | 156 | 213 | 42.28% | 40.83% | 42.28% | 7.72 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 597 | 267 | 330 | 44.72% | 44.17% | 44.79% | 5.28 pp | -63 | 38 | -1.66 |
| BTC Hourly | rf | RandomForest | 774 | 348 | 426 | 44.96% | 44.58% | 44.58% | 5.04 pp | -78 | 42 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 369 | 149 | 220 | 40.38% | 40.83% | 40.38% | 9.62 pp | -71 | 38 | -1.87 |
| BTC Hourly | nn | NN | 774 | 346 | 428 | 44.70% | 39.58% | 45.42% | 5.30 pp | -82 | 42 | -1.95 |
| BTC Daily | rf | RandomForest | 597 | 259 | 338 | 43.38% | 45.00% | 43.96% | 6.62 pp | -79 | 38 | -2.08 |
| BTC Market Hours Daily | rf | RandomForest | 423 | 171 | 252 | 40.43% | 39.58% | 40.43% | 9.57 pp | -81 | 38 | -2.13 |
| BTC Hourly | lstm | LSTM | 774 | 342 | 432 | 44.19% | 42.92% | 45.83% | 5.81 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 38.33% | 39.72% | 10.28 pp | -87 | 38 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 423 | 166 | 257 | 39.24% | 37.92% | 39.24% | 10.76 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 774 | 332 | 442 | 42.89% | 40.83% | 44.17% | 7.11 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 607 | 245 | 362 | 40.36% | 36.25% | 40.21% | 9.64 pp | -117 | 38 | -3.08 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 774 | 364 | 410 | 47.03% | 43.33% | 46.88% | 2.97 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 774 | 362 | 412 | 46.77% | 42.50% | 45.62% | 3.23 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 774 | 348 | 426 | 44.96% | 44.58% | 44.58% | 5.04 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 774 | 346 | 428 | 44.70% | 39.58% | 45.42% | 5.30 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 774 | 342 | 432 | 44.19% | 42.92% | 45.83% | 5.81 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 774 | 332 | 442 | 42.89% | 40.83% | 44.17% | 7.11 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 597 | 295 | 302 | 49.41% | 50.83% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 597 | 291 | 306 | 48.74% | 47.08% | 49.17% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Daily | nn | NN | 597 | 281 | 316 | 47.07% | 45.42% | 48.12% | 2.93 pp | -35 | 38 | -0.92 |
| BTC Daily | lstm | LSTM | 597 | 267 | 330 | 44.72% | 44.17% | 44.79% | 5.28 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 597 | 259 | 338 | 43.38% | 45.00% | 43.96% | 6.62 pp | -79 | 38 | -2.08 |
| BTC Daily | xgb | XGBoost | 607 | 245 | 362 | 40.36% | 36.25% | 40.21% | 9.64 pp | -117 | 38 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 369 | 181 | 188 | 49.05% | 48.75% | 49.05% | 0.95 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 369 | 174 | 195 | 47.15% | 45.42% | 47.15% | 2.85 pp | -21 | 38 | -0.55 |
| BTC Market Hours | nn | NN | 369 | 170 | 199 | 46.07% | 48.33% | 46.07% | 3.93 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 369 | 157 | 212 | 42.55% | 42.92% | 42.55% | 7.45 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 369 | 156 | 213 | 42.28% | 40.83% | 42.28% | 7.72 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 369 | 149 | 220 | 40.38% | 40.83% | 40.38% | 9.62 pp | -71 | 38 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 423 | 193 | 230 | 45.63% | 45.00% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 423 | 193 | 230 | 45.63% | 46.25% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 423 | 193 | 230 | 45.63% | 47.50% | 45.63% | 4.37 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 423 | 171 | 252 | 40.43% | 39.58% | 40.43% | 9.57 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 38.33% | 39.72% | 10.28 pp | -87 | 38 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 423 | 166 | 257 | 39.24% | 37.92% | 39.24% | 10.76 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
