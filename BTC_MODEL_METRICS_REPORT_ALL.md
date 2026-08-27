# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T03:27:37.691669+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 794 | 325 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 958 | 593 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 522 | 355 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 524 | 409 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 0 | 19 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 03:00:00+00:00 | 19 | 0 | 19 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 355 | 176 | 179 | 49.58% | 47.92% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Daily | transformer | Transformer | 583 | 289 | 294 | 49.57% | 52.08% | 49.79% | 0.43 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 583 | 287 | 296 | 49.23% | 48.75% | 49.58% | 0.77 pp | -9 | 37 | -0.24 |
| Consolidated Hourly | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 355 | 168 | 187 | 47.32% | 46.25% | 47.32% | 2.68 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 355 | 163 | 192 | 45.92% | 48.33% | 45.92% | 4.08 pp | -29 | 37 | -0.78 |
| BTC Daily | nn | NN | 583 | 275 | 308 | 47.17% | 45.42% | 47.92% | 2.83 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 409 | 188 | 221 | 45.97% | 46.25% | 45.97% | 4.03 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 409 | 188 | 221 | 45.97% | 47.50% | 45.97% | 4.03 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 409 | 187 | 222 | 45.72% | 47.50% | 45.72% | 4.28 pp | -35 | 37 | -0.95 |
| Consolidated Hourly | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 760 | 357 | 403 | 46.97% | 43.33% | 47.71% | 3.03 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 760 | 356 | 404 | 46.84% | 44.17% | 45.62% | 3.16 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 355 | 153 | 202 | 43.10% | 42.92% | 43.10% | 6.90 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 355 | 152 | 203 | 42.82% | 42.08% | 42.82% | 7.18 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 583 | 263 | 320 | 45.11% | 45.83% | 45.00% | 4.89 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 355 | 147 | 208 | 41.41% | 42.08% | 41.41% | 8.59 pp | -61 | 37 | -1.65 |
| BTC Hourly | nn | NN | 760 | 341 | 419 | 44.87% | 41.67% | 45.62% | 5.13 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 760 | 341 | 419 | 44.87% | 45.00% | 44.58% | 5.13 pp | -78 | 42 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 409 | 169 | 240 | 41.32% | 40.42% | 41.32% | 8.68 pp | -71 | 37 | -1.92 |
| BTC Daily | rf | RandomForest | 583 | 253 | 330 | 43.40% | 44.58% | 43.96% | 6.60 pp | -77 | 37 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 409 | 165 | 244 | 40.34% | 38.33% | 40.34% | 9.66 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 760 | 335 | 425 | 44.08% | 42.92% | 45.42% | 5.92 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 409 | 164 | 245 | 40.10% | 38.33% | 40.10% | 9.90 pp | -81 | 37 | -2.19 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| BTC Hourly | xgb | XGBoost | 760 | 327 | 433 | 43.03% | 42.08% | 44.58% | 6.97 pp | -106 | 42 | -2.52 |
| BTC Daily | xgb | XGBoost | 593 | 239 | 354 | 40.30% | 35.83% | 40.62% | 9.70 pp | -115 | 37 | -3.11 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 760 | 357 | 403 | 46.97% | 43.33% | 47.71% | 3.03 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 760 | 356 | 404 | 46.84% | 44.17% | 45.62% | 3.16 pp | -48 | 42 | -1.14 |
| BTC Hourly | nn | NN | 760 | 341 | 419 | 44.87% | 41.67% | 45.62% | 5.13 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 760 | 341 | 419 | 44.87% | 45.00% | 44.58% | 5.13 pp | -78 | 42 | -1.86 |
| BTC Hourly | lstm | LSTM | 760 | 335 | 425 | 44.08% | 42.92% | 45.42% | 5.92 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 760 | 327 | 433 | 43.03% | 42.08% | 44.58% | 6.97 pp | -106 | 42 | -2.52 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 583 | 289 | 294 | 49.57% | 52.08% | 49.79% | 0.43 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 583 | 287 | 296 | 49.23% | 48.75% | 49.58% | 0.77 pp | -9 | 37 | -0.24 |
| BTC Daily | nn | NN | 583 | 275 | 308 | 47.17% | 45.42% | 47.92% | 2.83 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 583 | 263 | 320 | 45.11% | 45.83% | 45.00% | 4.89 pp | -57 | 37 | -1.54 |
| BTC Daily | rf | RandomForest | 583 | 253 | 330 | 43.40% | 44.58% | 43.96% | 6.60 pp | -77 | 37 | -2.08 |
| BTC Daily | xgb | XGBoost | 593 | 239 | 354 | 40.30% | 35.83% | 40.62% | 9.70 pp | -115 | 37 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 355 | 176 | 179 | 49.58% | 47.92% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Market Hours | transformer | Transformer | 355 | 168 | 187 | 47.32% | 46.25% | 47.32% | 2.68 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 355 | 163 | 192 | 45.92% | 48.33% | 45.92% | 4.08 pp | -29 | 37 | -0.78 |
| BTC Market Hours | lstm | LSTM | 355 | 153 | 202 | 43.10% | 42.92% | 43.10% | 6.90 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 355 | 152 | 203 | 42.82% | 42.08% | 42.82% | 7.18 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 355 | 147 | 208 | 41.41% | 42.08% | 41.41% | 8.59 pp | -61 | 37 | -1.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 409 | 188 | 221 | 45.97% | 46.25% | 45.97% | 4.03 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 409 | 188 | 221 | 45.97% | 47.50% | 45.97% | 4.03 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 409 | 187 | 222 | 45.72% | 47.50% | 45.72% | 4.28 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 409 | 169 | 240 | 41.32% | 40.42% | 41.32% | 8.68 pp | -71 | 37 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 409 | 165 | 244 | 40.34% | 38.33% | 40.34% | 9.66 pp | -79 | 37 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 409 | 164 | 245 | 40.10% | 38.33% | 40.10% | 9.90 pp | -81 | 37 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 8 | 11 | 42.11% | 42.11% | 42.11% | 7.89 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

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
