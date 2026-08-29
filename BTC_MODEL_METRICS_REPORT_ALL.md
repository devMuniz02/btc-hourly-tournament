# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T19:35:56.426352+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1132 | 844 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1008 | 643 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 18:00:00+00:00 | 605 | 405 | 199 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 18:00:00+00:00 | 606 | 458 | 146 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 405 | 201 | 204 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 633 | 311 | 322 | 49.13% | 48.33% | 49.79% | 0.87 pp | -11 | 39 | -0.28 |
| Consolidated Hourly | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 633 | 309 | 324 | 48.82% | 46.25% | 50.21% | 1.18 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 405 | 192 | 213 | 47.41% | 51.25% | 47.41% | 2.59 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 405 | 188 | 217 | 46.42% | 42.92% | 46.42% | 3.58 pp | -29 | 41 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 458 | 212 | 246 | 46.29% | 46.25% | 46.29% | 3.71 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 458 | 210 | 248 | 45.85% | 47.08% | 45.85% | 4.15 pp | -38 | 40 | -0.95 |
| BTC Daily | nn | NN | 633 | 297 | 336 | 46.92% | 42.92% | 48.96% | 3.08 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 458 | 209 | 249 | 45.63% | 45.83% | 45.63% | 4.37 pp | -40 | 40 | -1.00 |
| BTC Hourly | transformer | Transformer | 810 | 382 | 428 | 47.16% | 45.00% | 46.04% | 2.84 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 810 | 381 | 429 | 47.04% | 44.17% | 46.88% | 2.96 pp | -48 | 44 | -1.09 |
| BTC Market Hours | lstm | LSTM | 405 | 179 | 226 | 44.20% | 45.83% | 44.20% | 5.80 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 405 | 174 | 231 | 42.96% | 42.08% | 42.96% | 7.04 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| BTC Hourly | nn | NN | 810 | 364 | 446 | 44.94% | 40.83% | 44.79% | 5.06 pp | -82 | 44 | -1.86 |
| BTC Daily | lstm | LSTM | 633 | 280 | 353 | 44.23% | 42.08% | 43.54% | 5.77 pp | -73 | 39 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 405 | 163 | 242 | 40.25% | 38.33% | 40.25% | 9.75 pp | -79 | 41 | -1.93 |
| BTC Hourly | rf | RandomForest | 810 | 362 | 448 | 44.69% | 44.17% | 44.38% | 5.31 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 458 | 188 | 270 | 41.05% | 40.83% | 41.05% | 8.95 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 458 | 187 | 271 | 40.83% | 40.00% | 40.83% | 9.17 pp | -84 | 40 | -2.10 |
| BTC Hourly | lstm | LSTM | 810 | 354 | 456 | 43.70% | 42.08% | 44.38% | 6.30 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 633 | 270 | 363 | 42.65% | 42.08% | 43.54% | 7.35 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 458 | 180 | 278 | 39.30% | 36.67% | 39.30% | 10.70 pp | -98 | 40 | -2.45 |
| BTC Hourly | xgb | XGBoost | 810 | 343 | 467 | 42.35% | 39.58% | 42.71% | 7.65 pp | -124 | 44 | -2.82 |
| Consolidated Hourly | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| BTC Daily | xgb | XGBoost | 643 | 252 | 391 | 39.19% | 31.25% | 39.38% | 10.81 pp | -139 | 39 | -3.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 810 | 382 | 428 | 47.16% | 45.00% | 46.04% | 2.84 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 810 | 381 | 429 | 47.04% | 44.17% | 46.88% | 2.96 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 810 | 364 | 446 | 44.94% | 40.83% | 44.79% | 5.06 pp | -82 | 44 | -1.86 |
| BTC Hourly | rf | RandomForest | 810 | 362 | 448 | 44.69% | 44.17% | 44.38% | 5.31 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 810 | 354 | 456 | 43.70% | 42.08% | 44.38% | 6.30 pp | -102 | 44 | -2.32 |
| BTC Hourly | xgb | XGBoost | 810 | 343 | 467 | 42.35% | 39.58% | 42.71% | 7.65 pp | -124 | 44 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 633 | 311 | 322 | 49.13% | 48.33% | 49.79% | 0.87 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 633 | 309 | 324 | 48.82% | 46.25% | 50.21% | 1.18 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 633 | 297 | 336 | 46.92% | 42.92% | 48.96% | 3.08 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 633 | 280 | 353 | 44.23% | 42.08% | 43.54% | 5.77 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 633 | 270 | 363 | 42.65% | 42.08% | 43.54% | 7.35 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 643 | 252 | 391 | 39.19% | 31.25% | 39.38% | 10.81 pp | -139 | 39 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 405 | 201 | 204 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 405 | 192 | 213 | 47.41% | 51.25% | 47.41% | 2.59 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 405 | 188 | 217 | 46.42% | 42.92% | 46.42% | 3.58 pp | -29 | 41 | -0.71 |
| BTC Market Hours | lstm | LSTM | 405 | 179 | 226 | 44.20% | 45.83% | 44.20% | 5.80 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 405 | 174 | 231 | 42.96% | 42.08% | 42.96% | 7.04 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 405 | 163 | 242 | 40.25% | 38.33% | 40.25% | 9.75 pp | -79 | 41 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 458 | 212 | 246 | 46.29% | 46.25% | 46.29% | 3.71 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 458 | 210 | 248 | 45.85% | 47.08% | 45.85% | 4.15 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 458 | 209 | 249 | 45.63% | 45.83% | 45.63% | 4.37 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 458 | 188 | 270 | 41.05% | 40.83% | 41.05% | 8.95 pp | -82 | 40 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 458 | 187 | 271 | 40.83% | 40.00% | 40.83% | 9.17 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 458 | 180 | 278 | 39.30% | 36.67% | 39.30% | 10.70 pp | -98 | 40 | -2.45 |

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
