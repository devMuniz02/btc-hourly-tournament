# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T06:20:39.973846+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 797 | 322 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 961 | 596 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 358 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 526 | 411 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 21 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 0 | 21 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 05:00:00+00:00 | 21 | 0 | 21 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 586 | 291 | 295 | 49.66% | 52.08% | 50.21% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 287 | 299 | 48.98% | 47.92% | 49.58% | 1.02 pp | -12 | 37 | -0.32 |
| Consolidated Hourly | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Daily | nn | NN | 586 | 277 | 309 | 47.27% | 45.83% | 48.33% | 2.73 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 411 | 189 | 222 | 45.99% | 47.08% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 411 | 189 | 222 | 45.99% | 48.33% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 411 | 188 | 223 | 45.74% | 46.25% | 45.74% | 4.26 pp | -35 | 37 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 411 | 169 | 242 | 41.12% | 40.42% | 41.12% | 8.88 pp | -73 | 37 | -1.97 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 411 | 165 | 246 | 40.15% | 38.33% | 40.15% | 9.85 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 37 | -2.24 |
| Consolidated Hourly | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 596 | 241 | 355 | 40.44% | 36.25% | 41.04% | 9.56 pp | -114 | 37 | -3.08 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 586 | 291 | 295 | 49.66% | 52.08% | 50.21% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 287 | 299 | 48.98% | 47.92% | 49.58% | 1.02 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 586 | 277 | 309 | 47.27% | 45.83% | 48.33% | 2.73 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 596 | 241 | 355 | 40.44% | 36.25% | 41.04% | 9.56 pp | -114 | 37 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 411 | 189 | 222 | 45.99% | 47.08% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 411 | 189 | 222 | 45.99% | 48.33% | 45.99% | 4.01 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 411 | 188 | 223 | 45.74% | 46.25% | 45.74% | 4.26 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 411 | 169 | 242 | 41.12% | 40.42% | 41.12% | 8.88 pp | -73 | 37 | -1.97 |
| BTC Market Hours Daily | xgb | XGBoost | 411 | 165 | 246 | 40.15% | 38.33% | 40.15% | 9.85 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 37 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 21 | 14 | 7 | 66.67% | 66.67% | 66.67% | 16.67 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 21 | 13 | 8 | 61.90% | 61.90% | 61.90% | 11.90 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 21 | 12 | 9 | 57.14% | 57.14% | 57.14% | 7.14 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 21 | 11 | 10 | 52.38% | 52.38% | 52.38% | 2.38 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 21 | 10 | 11 | 47.62% | 47.62% | 47.62% | 2.38 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 21 | 7 | 14 | 33.33% | 33.33% | 33.33% | 16.67 pp | -7 | 3 | -2.33 |

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
