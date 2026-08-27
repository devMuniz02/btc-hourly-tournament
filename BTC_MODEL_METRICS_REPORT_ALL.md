# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T21:21:28.727875+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 20:00:00+00:00 | 546 | 422 | 122 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 03:00:00+00:00 | 30 | 30 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 03:00:00+00:00 | 30 | 30 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 03:00:00+00:00 | 30 | 0 | 30 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 03:00:00+00:00 | 30 | 0 | 30 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 30 | 18 | 12 | 60.00% | 60.00% | 60.00% | 10.00 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 30 | 18 | 12 | 60.00% | 60.00% | 60.00% | 10.00 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Daily | transformer | Transformer | 597 | 295 | 302 | 49.41% | 50.83% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 369 | 181 | 188 | 49.05% | 48.75% | 49.05% | 0.95 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 597 | 291 | 306 | 48.74% | 47.08% | 49.17% | 1.26 pp | -15 | 38 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 369 | 174 | 195 | 47.15% | 45.42% | 47.15% | 2.85 pp | -21 | 38 | -0.55 |
| BTC Market Hours | nn | NN | 369 | 170 | 199 | 46.07% | 48.33% | 46.07% | 3.93 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 597 | 281 | 316 | 47.07% | 45.42% | 48.12% | 2.93 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 422 | 193 | 229 | 45.73% | 45.42% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | nn | NN | 422 | 193 | 229 | 45.73% | 46.67% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 422 | 192 | 230 | 45.50% | 47.08% | 45.50% | 4.50 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 774 | 364 | 410 | 47.03% | 43.33% | 46.88% | 2.97 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 774 | 362 | 412 | 46.77% | 42.50% | 45.62% | 3.23 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 369 | 157 | 212 | 42.55% | 42.92% | 42.55% | 7.45 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 369 | 156 | 213 | 42.28% | 40.83% | 42.28% | 7.72 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 597 | 267 | 330 | 44.72% | 44.17% | 44.79% | 5.28 pp | -63 | 38 | -1.66 |
| BTC Hourly | rf | RandomForest | 774 | 348 | 426 | 44.96% | 44.58% | 44.58% | 5.04 pp | -78 | 42 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 369 | 149 | 220 | 40.38% | 40.83% | 40.38% | 9.62 pp | -71 | 38 | -1.87 |
| BTC Hourly | nn | NN | 774 | 346 | 428 | 44.70% | 39.58% | 45.42% | 5.30 pp | -82 | 42 | -1.95 |
| BTC Daily | rf | RandomForest | 597 | 259 | 338 | 43.38% | 45.00% | 43.96% | 6.62 pp | -79 | 38 | -2.08 |
| BTC Market Hours Daily | rf | RandomForest | 422 | 171 | 251 | 40.52% | 40.00% | 40.52% | 9.48 pp | -80 | 38 | -2.11 |
| BTC Hourly | lstm | LSTM | 774 | 342 | 432 | 44.19% | 42.92% | 45.83% | 5.81 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 38.75% | 39.81% | 10.19 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 422 | 166 | 256 | 39.34% | 37.92% | 39.34% | 10.66 pp | -90 | 38 | -2.37 |
| BTC Hourly | xgb | XGBoost | 774 | 332 | 442 | 42.89% | 40.83% | 44.17% | 7.11 pp | -110 | 42 | -2.62 |
| Consolidated Hourly | nn | NN | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 607 | 245 | 362 | 40.36% | 36.25% | 40.21% | 9.64 pp | -117 | 38 | -3.08 |

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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 422 | 193 | 229 | 45.73% | 45.42% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | nn | NN | 422 | 193 | 229 | 45.73% | 46.67% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 422 | 192 | 230 | 45.50% | 47.08% | 45.50% | 4.50 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 422 | 171 | 251 | 40.52% | 40.00% | 40.52% | 9.48 pp | -80 | 38 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 38.75% | 39.81% | 10.19 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 422 | 166 | 256 | 39.34% | 37.92% | 39.34% | 10.66 pp | -90 | 38 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 30 | 18 | 12 | 60.00% | 60.00% | 60.00% | 10.00 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | nn | NN | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 30 | 18 | 12 | 60.00% | 60.00% | 60.00% | 10.00 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 30 | 17 | 13 | 56.67% | 56.67% | 56.67% | 6.67 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 30 | 15 | 15 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 4 | -3.00 |

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
