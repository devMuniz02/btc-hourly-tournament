# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T05:04:39.517664+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 796 | 323 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 960 | 595 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 524 | 357 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 410 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 585 | 291 | 294 | 49.74% | 52.08% | 50.21% | 0.26 pp | -3 | 37 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 357 | 177 | 180 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Daily | mlp_sklearn | MLPClassifier | 585 | 287 | 298 | 49.06% | 48.33% | 49.58% | 0.94 pp | -11 | 37 | -0.30 |
| BTC Market Hours | transformer | Transformer | 357 | 169 | 188 | 47.34% | 46.25% | 47.34% | 2.66 pp | -19 | 37 | -0.51 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Market Hours | nn | NN | 357 | 163 | 194 | 45.66% | 47.92% | 45.66% | 4.34 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | nn | NN | 410 | 189 | 221 | 46.10% | 47.50% | 46.10% | 3.90 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 585 | 276 | 309 | 47.18% | 45.42% | 48.12% | 2.82 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 410 | 188 | 222 | 45.85% | 46.25% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 410 | 188 | 222 | 45.85% | 47.92% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 762 | 358 | 404 | 46.98% | 43.75% | 47.50% | 3.02 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 762 | 356 | 406 | 46.72% | 43.75% | 45.62% | 3.28 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 357 | 155 | 202 | 43.42% | 43.33% | 43.42% | 6.58 pp | -47 | 37 | -1.27 |
| BTC Market Hours | rf | RandomForest | 357 | 153 | 204 | 42.86% | 42.08% | 42.86% | 7.14 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 585 | 264 | 321 | 45.13% | 45.83% | 45.21% | 4.87 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 357 | 147 | 210 | 41.18% | 42.08% | 41.18% | 8.82 pp | -63 | 37 | -1.70 |
| BTC Hourly | nn | NN | 762 | 341 | 421 | 44.75% | 41.25% | 45.42% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 762 | 341 | 421 | 44.75% | 45.00% | 44.38% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 410 | 169 | 241 | 41.22% | 40.42% | 41.22% | 8.78 pp | -72 | 37 | -1.95 |
| BTC Daily | rf | RandomForest | 585 | 253 | 332 | 43.25% | 44.17% | 43.96% | 6.75 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 762 | 336 | 426 | 44.09% | 43.33% | 45.42% | 5.91 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 410 | 165 | 245 | 40.24% | 38.33% | 40.24% | 9.76 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 37 | -2.22 |
| BTC Hourly | xgb | XGBoost | 762 | 327 | 435 | 42.91% | 42.08% | 44.17% | 7.09 pp | -108 | 42 | -2.57 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 595 | 240 | 355 | 40.34% | 35.83% | 40.83% | 9.66 pp | -115 | 37 | -3.11 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 762 | 358 | 404 | 46.98% | 43.75% | 47.50% | 3.02 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 762 | 356 | 406 | 46.72% | 43.75% | 45.62% | 3.28 pp | -50 | 42 | -1.19 |
| BTC Hourly | nn | NN | 762 | 341 | 421 | 44.75% | 41.25% | 45.42% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 762 | 341 | 421 | 44.75% | 45.00% | 44.38% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 762 | 336 | 426 | 44.09% | 43.33% | 45.42% | 5.91 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 762 | 327 | 435 | 42.91% | 42.08% | 44.17% | 7.09 pp | -108 | 42 | -2.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 585 | 291 | 294 | 49.74% | 52.08% | 50.21% | 0.26 pp | -3 | 37 | -0.08 |
| BTC Daily | mlp_sklearn | MLPClassifier | 585 | 287 | 298 | 49.06% | 48.33% | 49.58% | 0.94 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 585 | 276 | 309 | 47.18% | 45.42% | 48.12% | 2.82 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 585 | 264 | 321 | 45.13% | 45.83% | 45.21% | 4.87 pp | -57 | 37 | -1.54 |
| BTC Daily | rf | RandomForest | 585 | 253 | 332 | 43.25% | 44.17% | 43.96% | 6.75 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 595 | 240 | 355 | 40.34% | 35.83% | 40.83% | 9.66 pp | -115 | 37 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 357 | 177 | 180 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Market Hours | transformer | Transformer | 357 | 169 | 188 | 47.34% | 46.25% | 47.34% | 2.66 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 357 | 163 | 194 | 45.66% | 47.92% | 45.66% | 4.34 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 357 | 155 | 202 | 43.42% | 43.33% | 43.42% | 6.58 pp | -47 | 37 | -1.27 |
| BTC Market Hours | rf | RandomForest | 357 | 153 | 204 | 42.86% | 42.08% | 42.86% | 7.14 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 357 | 147 | 210 | 41.18% | 42.08% | 41.18% | 8.82 pp | -63 | 37 | -1.70 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 410 | 189 | 221 | 46.10% | 47.50% | 46.10% | 3.90 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 410 | 188 | 222 | 45.85% | 46.25% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 410 | 188 | 222 | 45.85% | 47.92% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 410 | 169 | 241 | 41.22% | 40.42% | 41.22% | 8.78 pp | -72 | 37 | -1.95 |
| BTC Market Hours Daily | xgb | XGBoost | 410 | 165 | 245 | 40.24% | 38.33% | 40.24% | 9.76 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 37 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

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
