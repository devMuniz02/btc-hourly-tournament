# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T13:54:46.793045+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 821 | 298 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 985 | 620 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 12:00:00+00:00 | 563 | 382 | 180 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 12:00:00+00:00 | 564 | 435 | 127 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 03:00:00+00:00 | 41 | 41 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 03:00:00+00:00 | 41 | 41 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 03:00:00+00:00 | 41 | 0 | 41 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 03:00:00+00:00 | 41 | 0 | 41 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 41 | 24 | 17 | 58.54% | 58.54% | 58.54% | 8.54 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 41 | 24 | 17 | 58.54% | 58.54% | 58.54% | 8.54 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 5 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 382 | 187 | 195 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 610 | 299 | 311 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 610 | 299 | 311 | 49.02% | 49.58% | 50.00% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Market Hours | transformer | Transformer | 382 | 178 | 204 | 46.60% | 43.75% | 46.60% | 3.40 pp | -26 | 39 | -0.67 |
| BTC Market Hours | nn | NN | 382 | 177 | 205 | 46.34% | 48.75% | 46.34% | 3.66 pp | -28 | 39 | -0.72 |
| BTC Daily | nn | NN | 610 | 287 | 323 | 47.05% | 44.17% | 48.54% | 2.95 pp | -36 | 38 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 787 | 372 | 415 | 47.27% | 44.58% | 47.29% | 2.73 pp | -43 | 43 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 435 | 198 | 237 | 45.52% | 45.42% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 435 | 198 | 237 | 45.52% | 46.67% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | nn | NN | 435 | 197 | 238 | 45.29% | 45.83% | 45.29% | 4.71 pp | -41 | 39 | -1.05 |
| BTC Hourly | transformer | Transformer | 787 | 369 | 418 | 46.89% | 43.33% | 46.04% | 3.11 pp | -49 | 43 | -1.14 |
| BTC Market Hours | lstm | LSTM | 382 | 163 | 219 | 42.67% | 43.33% | 42.67% | 7.33 pp | -56 | 39 | -1.44 |
| BTC Market Hours | rf | RandomForest | 382 | 162 | 220 | 42.41% | 40.42% | 42.41% | 7.59 pp | -58 | 39 | -1.49 |
| BTC Daily | lstm | LSTM | 610 | 272 | 338 | 44.59% | 43.75% | 44.58% | 5.41 pp | -66 | 38 | -1.74 |
| BTC Hourly | nn | NN | 787 | 354 | 433 | 44.98% | 40.42% | 45.83% | 5.02 pp | -79 | 43 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 382 | 153 | 229 | 40.05% | 38.33% | 40.05% | 9.95 pp | -76 | 39 | -1.95 |
| BTC Hourly | rf | RandomForest | 787 | 350 | 437 | 44.47% | 42.92% | 43.96% | 5.53 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 787 | 347 | 440 | 44.09% | 43.75% | 45.42% | 5.91 pp | -93 | 43 | -2.16 |
| BTC Market Hours Daily | rf | RandomForest | 435 | 175 | 260 | 40.23% | 38.75% | 40.23% | 9.77 pp | -85 | 39 | -2.18 |
| BTC Daily | rf | RandomForest | 610 | 263 | 347 | 43.11% | 43.33% | 43.75% | 6.89 pp | -84 | 38 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 435 | 172 | 263 | 39.54% | 37.92% | 39.54% | 10.46 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 435 | 170 | 265 | 39.08% | 37.92% | 39.08% | 10.92 pp | -95 | 39 | -2.44 |
| Consolidated Hourly | nn | NN | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 5 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 5 | -2.60 |
| BTC Hourly | xgb | XGBoost | 787 | 336 | 451 | 42.69% | 40.00% | 44.17% | 7.31 pp | -115 | 43 | -2.67 |
| BTC Daily | xgb | XGBoost | 620 | 247 | 373 | 39.84% | 33.75% | 40.00% | 10.16 pp | -126 | 38 | -3.32 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 787 | 372 | 415 | 47.27% | 44.58% | 47.29% | 2.73 pp | -43 | 43 | -1.00 |
| BTC Hourly | transformer | Transformer | 787 | 369 | 418 | 46.89% | 43.33% | 46.04% | 3.11 pp | -49 | 43 | -1.14 |
| BTC Hourly | nn | NN | 787 | 354 | 433 | 44.98% | 40.42% | 45.83% | 5.02 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 787 | 350 | 437 | 44.47% | 42.92% | 43.96% | 5.53 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 787 | 347 | 440 | 44.09% | 43.75% | 45.42% | 5.91 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 787 | 336 | 451 | 42.69% | 40.00% | 44.17% | 7.31 pp | -115 | 43 | -2.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 610 | 299 | 311 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 610 | 299 | 311 | 49.02% | 49.58% | 50.00% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 610 | 287 | 323 | 47.05% | 44.17% | 48.54% | 2.95 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 610 | 272 | 338 | 44.59% | 43.75% | 44.58% | 5.41 pp | -66 | 38 | -1.74 |
| BTC Daily | rf | RandomForest | 610 | 263 | 347 | 43.11% | 43.33% | 43.75% | 6.89 pp | -84 | 38 | -2.21 |
| BTC Daily | xgb | XGBoost | 620 | 247 | 373 | 39.84% | 33.75% | 40.00% | 10.16 pp | -126 | 38 | -3.32 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 382 | 187 | 195 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Market Hours | transformer | Transformer | 382 | 178 | 204 | 46.60% | 43.75% | 46.60% | 3.40 pp | -26 | 39 | -0.67 |
| BTC Market Hours | nn | NN | 382 | 177 | 205 | 46.34% | 48.75% | 46.34% | 3.66 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 382 | 163 | 219 | 42.67% | 43.33% | 42.67% | 7.33 pp | -56 | 39 | -1.44 |
| BTC Market Hours | rf | RandomForest | 382 | 162 | 220 | 42.41% | 40.42% | 42.41% | 7.59 pp | -58 | 39 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 382 | 153 | 229 | 40.05% | 38.33% | 40.05% | 9.95 pp | -76 | 39 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 435 | 198 | 237 | 45.52% | 45.42% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 435 | 198 | 237 | 45.52% | 46.67% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 435 | 197 | 238 | 45.29% | 45.83% | 45.29% | 4.71 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 435 | 175 | 260 | 40.23% | 38.75% | 40.23% | 9.77 pp | -85 | 39 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 435 | 172 | 263 | 39.54% | 37.92% | 39.54% | 10.46 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 435 | 170 | 265 | 39.08% | 37.92% | 39.08% | 10.92 pp | -95 | 39 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 41 | 24 | 17 | 58.54% | 58.54% | 58.54% | 8.54 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 5 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 41 | 24 | 17 | 58.54% | 58.54% | 58.54% | 8.54 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 41 | 22 | 19 | 53.66% | 53.66% | 53.66% | 3.66 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 41 | 14 | 27 | 34.15% | 34.15% | 34.15% | 15.85 pp | -13 | 5 | -2.60 |

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
