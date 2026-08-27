# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T16:49:13.165890+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 805 | 314 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 968 | 603 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 536 | 365 | 170 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 538 | 419 | 117 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 365 | 180 | 185 | 49.32% | 48.75% | 49.32% | 0.68 pp | -5 | 37 | -0.14 |
| BTC Daily | transformer | Transformer | 593 | 293 | 300 | 49.41% | 51.25% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 593 | 289 | 304 | 48.74% | 46.67% | 49.17% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Market Hours | transformer | Transformer | 365 | 173 | 192 | 47.40% | 46.67% | 47.40% | 2.60 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 365 | 167 | 198 | 45.75% | 47.92% | 45.75% | 4.25 pp | -31 | 37 | -0.84 |
| BTC Daily | nn | NN | 593 | 279 | 314 | 47.05% | 45.00% | 48.12% | 2.95 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 419 | 192 | 227 | 45.82% | 45.83% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | nn | NN | 419 | 192 | 227 | 45.82% | 47.08% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 419 | 191 | 228 | 45.58% | 47.08% | 45.58% | 4.42 pp | -37 | 37 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 771 | 363 | 408 | 47.08% | 43.33% | 47.29% | 2.92 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 771 | 361 | 410 | 46.82% | 42.92% | 45.62% | 3.18 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 365 | 157 | 208 | 43.01% | 43.33% | 43.01% | 6.99 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 365 | 155 | 210 | 42.47% | 41.67% | 42.47% | 7.53 pp | -55 | 37 | -1.49 |
| BTC Daily | lstm | LSTM | 593 | 266 | 327 | 44.86% | 44.58% | 45.00% | 5.14 pp | -61 | 38 | -1.61 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 365 | 149 | 216 | 40.82% | 41.67% | 40.82% | 9.18 pp | -67 | 37 | -1.81 |
| BTC Hourly | rf | RandomForest | 771 | 346 | 425 | 44.88% | 45.00% | 44.58% | 5.12 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 771 | 344 | 427 | 44.62% | 39.58% | 45.42% | 5.38 pp | -83 | 42 | -1.98 |
| BTC Daily | rf | RandomForest | 593 | 256 | 337 | 43.17% | 44.17% | 43.54% | 6.83 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | rf | RandomForest | 419 | 170 | 249 | 40.57% | 40.00% | 40.57% | 9.43 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 771 | 340 | 431 | 44.10% | 42.92% | 45.42% | 5.90 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 38.75% | 39.86% | 10.14 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 419 | 166 | 253 | 39.62% | 37.92% | 39.62% | 10.38 pp | -87 | 37 | -2.35 |
| BTC Hourly | xgb | XGBoost | 771 | 331 | 440 | 42.93% | 41.25% | 44.38% | 7.07 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 603 | 242 | 361 | 40.13% | 35.83% | 40.00% | 9.87 pp | -119 | 38 | -3.13 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 771 | 363 | 408 | 47.08% | 43.33% | 47.29% | 2.92 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 771 | 361 | 410 | 46.82% | 42.92% | 45.62% | 3.18 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 771 | 346 | 425 | 44.88% | 45.00% | 44.58% | 5.12 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 771 | 344 | 427 | 44.62% | 39.58% | 45.42% | 5.38 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 771 | 340 | 431 | 44.10% | 42.92% | 45.42% | 5.90 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 771 | 331 | 440 | 42.93% | 41.25% | 44.38% | 7.07 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 593 | 293 | 300 | 49.41% | 51.25% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 593 | 289 | 304 | 48.74% | 46.67% | 49.17% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Daily | nn | NN | 593 | 279 | 314 | 47.05% | 45.00% | 48.12% | 2.95 pp | -35 | 38 | -0.92 |
| BTC Daily | lstm | LSTM | 593 | 266 | 327 | 44.86% | 44.58% | 45.00% | 5.14 pp | -61 | 38 | -1.61 |
| BTC Daily | rf | RandomForest | 593 | 256 | 337 | 43.17% | 44.17% | 43.54% | 6.83 pp | -81 | 38 | -2.13 |
| BTC Daily | xgb | XGBoost | 603 | 242 | 361 | 40.13% | 35.83% | 40.00% | 9.87 pp | -119 | 38 | -3.13 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 365 | 180 | 185 | 49.32% | 48.75% | 49.32% | 0.68 pp | -5 | 37 | -0.14 |
| BTC Market Hours | transformer | Transformer | 365 | 173 | 192 | 47.40% | 46.67% | 47.40% | 2.60 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 365 | 167 | 198 | 45.75% | 47.92% | 45.75% | 4.25 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 365 | 157 | 208 | 43.01% | 43.33% | 43.01% | 6.99 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 365 | 155 | 210 | 42.47% | 41.67% | 42.47% | 7.53 pp | -55 | 37 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 365 | 149 | 216 | 40.82% | 41.67% | 40.82% | 9.18 pp | -67 | 37 | -1.81 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 419 | 192 | 227 | 45.82% | 45.83% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | nn | NN | 419 | 192 | 227 | 45.82% | 47.08% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 419 | 191 | 228 | 45.58% | 47.08% | 45.58% | 4.42 pp | -37 | 37 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 419 | 170 | 249 | 40.57% | 40.00% | 40.57% | 9.43 pp | -79 | 37 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 38.75% | 39.86% | 10.14 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 419 | 166 | 253 | 39.62% | 37.92% | 39.62% | 10.38 pp | -87 | 37 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

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
