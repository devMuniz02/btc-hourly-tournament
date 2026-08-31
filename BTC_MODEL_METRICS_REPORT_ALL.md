# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T03:41:35.230385+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1157 | 869 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1033 | 668 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 649 | 430 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 650 | 483 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 658 | 343 | 315 | 52.13% | 50.00% | 53.12% | 2.13 pp | 28 | 40 | 0.70 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 430 | 219 | 211 | 50.93% | 47.92% | 50.93% | 0.93 pp | 8 | 42 | 0.19 |
| BTC Market Hours | nn | NN | 430 | 212 | 218 | 49.30% | 50.42% | 49.30% | 0.70 pp | -6 | 42 | -0.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 835 | 414 | 421 | 49.58% | 47.92% | 49.79% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Daily | nn | NN | 658 | 317 | 341 | 48.18% | 46.25% | 50.00% | 1.82 pp | -24 | 40 | -0.60 |
| BTC Market Hours Daily | nn | NN | 483 | 226 | 257 | 46.79% | 44.58% | 47.08% | 3.21 pp | -31 | 42 | -0.74 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| BTC Daily | transformer | Transformer | 658 | 311 | 347 | 47.26% | 44.58% | 48.96% | 2.74 pp | -36 | 40 | -0.90 |
| BTC Hourly | transformer | Transformer | 835 | 397 | 438 | 47.54% | 46.67% | 45.42% | 2.46 pp | -41 | 45 | -0.91 |
| BTC Hourly | nn | NN | 835 | 396 | 439 | 47.43% | 47.08% | 46.67% | 2.57 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 221 | 262 | 45.76% | 47.08% | 46.04% | 4.24 pp | -41 | 42 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 221 | 262 | 45.76% | 42.92% | 45.83% | 4.24 pp | -41 | 42 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours | transformer | Transformer | 430 | 194 | 236 | 45.12% | 40.83% | 45.12% | 4.88 pp | -42 | 42 | -1.00 |
| BTC Hourly | rf | RandomForest | 835 | 392 | 443 | 46.95% | 45.83% | 45.83% | 3.05 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 430 | 189 | 241 | 43.95% | 44.17% | 43.95% | 6.05 pp | -52 | 42 | -1.24 |
| BTC Daily | lstm | LSTM | 658 | 303 | 355 | 46.05% | 40.42% | 45.83% | 3.95 pp | -52 | 40 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 212 | 271 | 43.89% | 43.33% | 43.96% | 6.11 pp | -59 | 42 | -1.40 |
| BTC Market Hours | lstm | LSTM | 430 | 184 | 246 | 42.79% | 40.83% | 42.79% | 7.21 pp | -62 | 42 | -1.48 |
| BTC Daily | rf | RandomForest | 658 | 298 | 360 | 45.29% | 42.50% | 46.25% | 4.71 pp | -62 | 40 | -1.55 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 200 | 283 | 41.41% | 39.17% | 41.46% | 8.59 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 835 | 373 | 462 | 44.67% | 43.33% | 45.00% | 5.33 pp | -89 | 45 | -1.98 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 430 | 173 | 257 | 40.23% | 38.33% | 40.23% | 9.77 pp | -84 | 42 | -2.00 |
| BTC Hourly | xgb | XGBoost | 835 | 368 | 467 | 44.07% | 43.33% | 43.96% | 5.93 pp | -99 | 45 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 194 | 289 | 40.17% | 36.67% | 40.21% | 9.83 pp | -95 | 42 | -2.26 |
| BTC Daily | xgb | XGBoost | 668 | 270 | 398 | 40.42% | 35.00% | 41.04% | 9.58 pp | -128 | 40 | -3.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 835 | 414 | 421 | 49.58% | 47.92% | 49.79% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Hourly | transformer | Transformer | 835 | 397 | 438 | 47.54% | 46.67% | 45.42% | 2.46 pp | -41 | 45 | -0.91 |
| BTC Hourly | nn | NN | 835 | 396 | 439 | 47.43% | 47.08% | 46.67% | 2.57 pp | -43 | 45 | -0.96 |
| BTC Hourly | rf | RandomForest | 835 | 392 | 443 | 46.95% | 45.83% | 45.83% | 3.05 pp | -51 | 45 | -1.13 |
| BTC Hourly | lstm | LSTM | 835 | 373 | 462 | 44.67% | 43.33% | 45.00% | 5.33 pp | -89 | 45 | -1.98 |
| BTC Hourly | xgb | XGBoost | 835 | 368 | 467 | 44.07% | 43.33% | 43.96% | 5.93 pp | -99 | 45 | -2.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 658 | 343 | 315 | 52.13% | 50.00% | 53.12% | 2.13 pp | 28 | 40 | 0.70 |
| BTC Daily | nn | NN | 658 | 317 | 341 | 48.18% | 46.25% | 50.00% | 1.82 pp | -24 | 40 | -0.60 |
| BTC Daily | transformer | Transformer | 658 | 311 | 347 | 47.26% | 44.58% | 48.96% | 2.74 pp | -36 | 40 | -0.90 |
| BTC Daily | lstm | LSTM | 658 | 303 | 355 | 46.05% | 40.42% | 45.83% | 3.95 pp | -52 | 40 | -1.30 |
| BTC Daily | rf | RandomForest | 658 | 298 | 360 | 45.29% | 42.50% | 46.25% | 4.71 pp | -62 | 40 | -1.55 |
| BTC Daily | xgb | XGBoost | 668 | 270 | 398 | 40.42% | 35.00% | 41.04% | 9.58 pp | -128 | 40 | -3.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 430 | 219 | 211 | 50.93% | 47.92% | 50.93% | 0.93 pp | 8 | 42 | 0.19 |
| BTC Market Hours | nn | NN | 430 | 212 | 218 | 49.30% | 50.42% | 49.30% | 0.70 pp | -6 | 42 | -0.14 |
| BTC Market Hours | transformer | Transformer | 430 | 194 | 236 | 45.12% | 40.83% | 45.12% | 4.88 pp | -42 | 42 | -1.00 |
| BTC Market Hours | rf | RandomForest | 430 | 189 | 241 | 43.95% | 44.17% | 43.95% | 6.05 pp | -52 | 42 | -1.24 |
| BTC Market Hours | lstm | LSTM | 430 | 184 | 246 | 42.79% | 40.83% | 42.79% | 7.21 pp | -62 | 42 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 430 | 173 | 257 | 40.23% | 38.33% | 40.23% | 9.77 pp | -84 | 42 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 483 | 226 | 257 | 46.79% | 44.58% | 47.08% | 3.21 pp | -31 | 42 | -0.74 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 221 | 262 | 45.76% | 47.08% | 46.04% | 4.24 pp | -41 | 42 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 221 | 262 | 45.76% | 42.92% | 45.83% | 4.24 pp | -41 | 42 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 212 | 271 | 43.89% | 43.33% | 43.96% | 6.11 pp | -59 | 42 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 200 | 283 | 41.41% | 39.17% | 41.46% | 8.59 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 194 | 289 | 40.17% | 36.67% | 40.21% | 9.83 pp | -95 | 42 | -2.26 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

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
