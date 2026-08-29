# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T12:01:22.922156+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1126 | 838 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1002 | 637 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 592 | 399 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 594 | 453 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 07:00:00+00:00 | 56 | 56 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 07:00:00+00:00 | 56 | 56 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 07:00:00+00:00 | 56 | 0 | 56 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 07:00:00+00:00 | 56 | 0 | 56 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 399 | 197 | 202 | 49.37% | 47.92% | 49.37% | 0.63 pp | -5 | 40 | -0.12 |
| BTC Daily | transformer | Transformer | 627 | 309 | 318 | 49.28% | 48.33% | 50.21% | 0.72 pp | -9 | 39 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 627 | 306 | 321 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 399 | 188 | 211 | 47.12% | 50.00% | 47.12% | 2.88 pp | -23 | 40 | -0.57 |
| BTC Market Hours | transformer | Transformer | 399 | 185 | 214 | 46.37% | 42.50% | 46.37% | 3.63 pp | -29 | 40 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 453 | 209 | 244 | 46.14% | 47.50% | 46.14% | 3.86 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 453 | 208 | 245 | 45.92% | 45.00% | 45.92% | 4.08 pp | -37 | 40 | -0.93 |
| BTC Daily | nn | NN | 627 | 295 | 332 | 47.05% | 43.75% | 48.96% | 2.95 pp | -37 | 39 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 804 | 380 | 424 | 47.26% | 44.58% | 47.08% | 2.74 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | nn | NN | 453 | 206 | 247 | 45.47% | 45.83% | 45.47% | 4.53 pp | -41 | 40 | -1.02 |
| BTC Hourly | transformer | Transformer | 804 | 379 | 425 | 47.14% | 45.00% | 46.46% | 2.86 pp | -46 | 43 | -1.07 |
| BTC Market Hours | lstm | LSTM | 399 | 175 | 224 | 43.86% | 44.58% | 43.86% | 6.14 pp | -49 | 40 | -1.23 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| BTC Market Hours | rf | RandomForest | 399 | 170 | 229 | 42.61% | 41.25% | 42.61% | 7.39 pp | -59 | 40 | -1.48 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 627 | 279 | 348 | 44.50% | 42.92% | 44.17% | 5.50 pp | -69 | 39 | -1.77 |
| BTC Hourly | nn | NN | 804 | 362 | 442 | 45.02% | 40.83% | 45.00% | 4.98 pp | -80 | 43 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 399 | 161 | 238 | 40.35% | 38.33% | 40.35% | 9.65 pp | -77 | 40 | -1.93 |
| BTC Hourly | rf | RandomForest | 804 | 359 | 445 | 44.65% | 44.17% | 44.38% | 5.35 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 453 | 184 | 269 | 40.62% | 40.00% | 40.62% | 9.38 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 453 | 183 | 270 | 40.40% | 38.75% | 40.40% | 9.60 pp | -87 | 40 | -2.17 |
| BTC Hourly | lstm | LSTM | 804 | 353 | 451 | 43.91% | 43.33% | 45.00% | 6.09 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 627 | 269 | 358 | 42.90% | 42.92% | 43.75% | 7.10 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 453 | 178 | 275 | 39.29% | 37.08% | 39.29% | 10.71 pp | -97 | 40 | -2.42 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 804 | 341 | 463 | 42.41% | 39.58% | 43.54% | 7.59 pp | -122 | 43 | -2.84 |
| BTC Daily | xgb | XGBoost | 637 | 251 | 386 | 39.40% | 32.08% | 39.79% | 10.60 pp | -135 | 39 | -3.46 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 804 | 380 | 424 | 47.26% | 44.58% | 47.08% | 2.74 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 804 | 379 | 425 | 47.14% | 45.00% | 46.46% | 2.86 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 804 | 362 | 442 | 45.02% | 40.83% | 45.00% | 4.98 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 804 | 359 | 445 | 44.65% | 44.17% | 44.38% | 5.35 pp | -86 | 43 | -2.00 |
| BTC Hourly | lstm | LSTM | 804 | 353 | 451 | 43.91% | 43.33% | 45.00% | 6.09 pp | -98 | 43 | -2.28 |
| BTC Hourly | xgb | XGBoost | 804 | 341 | 463 | 42.41% | 39.58% | 43.54% | 7.59 pp | -122 | 43 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 627 | 309 | 318 | 49.28% | 48.33% | 50.21% | 0.72 pp | -9 | 39 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 627 | 306 | 321 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 627 | 295 | 332 | 47.05% | 43.75% | 48.96% | 2.95 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 627 | 279 | 348 | 44.50% | 42.92% | 44.17% | 5.50 pp | -69 | 39 | -1.77 |
| BTC Daily | rf | RandomForest | 627 | 269 | 358 | 42.90% | 42.92% | 43.75% | 7.10 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 637 | 251 | 386 | 39.40% | 32.08% | 39.79% | 10.60 pp | -135 | 39 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 399 | 197 | 202 | 49.37% | 47.92% | 49.37% | 0.63 pp | -5 | 40 | -0.12 |
| BTC Market Hours | nn | NN | 399 | 188 | 211 | 47.12% | 50.00% | 47.12% | 2.88 pp | -23 | 40 | -0.57 |
| BTC Market Hours | transformer | Transformer | 399 | 185 | 214 | 46.37% | 42.50% | 46.37% | 3.63 pp | -29 | 40 | -0.72 |
| BTC Market Hours | lstm | LSTM | 399 | 175 | 224 | 43.86% | 44.58% | 43.86% | 6.14 pp | -49 | 40 | -1.23 |
| BTC Market Hours | rf | RandomForest | 399 | 170 | 229 | 42.61% | 41.25% | 42.61% | 7.39 pp | -59 | 40 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 399 | 161 | 238 | 40.35% | 38.33% | 40.35% | 9.65 pp | -77 | 40 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 453 | 209 | 244 | 46.14% | 47.50% | 46.14% | 3.86 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 453 | 208 | 245 | 45.92% | 45.00% | 45.92% | 4.08 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 453 | 206 | 247 | 45.47% | 45.83% | 45.47% | 4.53 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 453 | 184 | 269 | 40.62% | 40.00% | 40.62% | 9.38 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 453 | 183 | 270 | 40.40% | 38.75% | 40.40% | 9.60 pp | -87 | 40 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 453 | 178 | 275 | 39.29% | 37.08% | 39.29% | 10.71 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

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
