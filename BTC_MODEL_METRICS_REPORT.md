# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T16:23:16.900749+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 823 | 296 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 986 | 621 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 567 | 383 | 183 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 15:00:00+00:00 | 569 | 437 | 130 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 43 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 0 | 43 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 05:00:00+00:00 | 43 | 0 | 43 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 383 | 188 | 195 | 49.09% | 47.50% | 49.09% | 0.91 pp | -7 | 39 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 611 | 300 | 311 | 49.10% | 49.58% | 50.21% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 611 | 299 | 312 | 48.94% | 47.08% | 50.00% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 383 | 178 | 205 | 46.48% | 43.33% | 46.48% | 3.52 pp | -27 | 39 | -0.69 |
| BTC Market Hours | nn | NN | 383 | 177 | 206 | 46.21% | 48.75% | 46.21% | 3.79 pp | -29 | 39 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 437 | 200 | 237 | 45.77% | 47.50% | 45.77% | 4.23 pp | -37 | 39 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 789 | 373 | 416 | 47.28% | 45.00% | 47.29% | 2.72 pp | -43 | 43 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 437 | 199 | 238 | 45.54% | 45.83% | 45.54% | 4.46 pp | -39 | 39 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 611 | 286 | 325 | 46.81% | 43.75% | 48.33% | 3.19 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 437 | 198 | 239 | 45.31% | 46.25% | 45.31% | 4.69 pp | -41 | 39 | -1.05 |
| BTC Hourly | transformer | Transformer | 789 | 371 | 418 | 47.02% | 44.17% | 46.46% | 2.98 pp | -47 | 43 | -1.09 |
| BTC Market Hours | lstm | LSTM | 383 | 164 | 219 | 42.82% | 43.33% | 42.82% | 7.18 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 383 | 163 | 220 | 42.56% | 40.83% | 42.56% | 7.44 pp | -57 | 39 | -1.46 |
| BTC Daily | lstm | LSTM | 611 | 274 | 337 | 44.84% | 44.58% | 45.00% | 5.16 pp | -63 | 38 | -1.66 |
| BTC Hourly | nn | NN | 789 | 356 | 433 | 45.12% | 40.83% | 45.83% | 4.88 pp | -77 | 43 | -1.79 |
| BTC Market Hours | xgb | XGBoost | 383 | 154 | 229 | 40.21% | 38.33% | 40.21% | 9.79 pp | -75 | 39 | -1.92 |
| BTC Hourly | rf | RandomForest | 789 | 350 | 439 | 44.36% | 42.50% | 43.75% | 5.64 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 437 | 177 | 260 | 40.50% | 38.75% | 40.50% | 9.50 pp | -83 | 39 | -2.13 |
| BTC Hourly | lstm | LSTM | 789 | 348 | 441 | 44.11% | 44.17% | 45.42% | 5.89 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 611 | 262 | 349 | 42.88% | 42.92% | 43.54% | 7.12 pp | -87 | 38 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 437 | 172 | 265 | 39.36% | 37.08% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 437 | 172 | 265 | 39.36% | 37.92% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |
| BTC Hourly | xgb | XGBoost | 789 | 336 | 453 | 42.59% | 39.17% | 43.96% | 7.41 pp | -117 | 43 | -2.72 |
| BTC Daily | xgb | XGBoost | 621 | 246 | 375 | 39.61% | 33.33% | 39.79% | 10.39 pp | -129 | 38 | -3.39 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 789 | 373 | 416 | 47.28% | 45.00% | 47.29% | 2.72 pp | -43 | 43 | -1.00 |
| BTC Hourly | transformer | Transformer | 789 | 371 | 418 | 47.02% | 44.17% | 46.46% | 2.98 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 789 | 356 | 433 | 45.12% | 40.83% | 45.83% | 4.88 pp | -77 | 43 | -1.79 |
| BTC Hourly | rf | RandomForest | 789 | 350 | 439 | 44.36% | 42.50% | 43.75% | 5.64 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 789 | 348 | 441 | 44.11% | 44.17% | 45.42% | 5.89 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 789 | 336 | 453 | 42.59% | 39.17% | 43.96% | 7.41 pp | -117 | 43 | -2.72 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 611 | 300 | 311 | 49.10% | 49.58% | 50.21% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 611 | 299 | 312 | 48.94% | 47.08% | 50.00% | 1.06 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 611 | 286 | 325 | 46.81% | 43.75% | 48.33% | 3.19 pp | -39 | 38 | -1.03 |
| BTC Daily | lstm | LSTM | 611 | 274 | 337 | 44.84% | 44.58% | 45.00% | 5.16 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 611 | 262 | 349 | 42.88% | 42.92% | 43.54% | 7.12 pp | -87 | 38 | -2.29 |
| BTC Daily | xgb | XGBoost | 621 | 246 | 375 | 39.61% | 33.33% | 39.79% | 10.39 pp | -129 | 38 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 383 | 188 | 195 | 49.09% | 47.50% | 49.09% | 0.91 pp | -7 | 39 | -0.18 |
| BTC Market Hours | transformer | Transformer | 383 | 178 | 205 | 46.48% | 43.33% | 46.48% | 3.52 pp | -27 | 39 | -0.69 |
| BTC Market Hours | nn | NN | 383 | 177 | 206 | 46.21% | 48.75% | 46.21% | 3.79 pp | -29 | 39 | -0.74 |
| BTC Market Hours | lstm | LSTM | 383 | 164 | 219 | 42.82% | 43.33% | 42.82% | 7.18 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 383 | 163 | 220 | 42.56% | 40.83% | 42.56% | 7.44 pp | -57 | 39 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 383 | 154 | 229 | 40.21% | 38.33% | 40.21% | 9.79 pp | -75 | 39 | -1.92 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 437 | 200 | 237 | 45.77% | 47.50% | 45.77% | 4.23 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 437 | 199 | 238 | 45.54% | 45.83% | 45.54% | 4.46 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 437 | 198 | 239 | 45.31% | 46.25% | 45.31% | 4.69 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 437 | 177 | 260 | 40.50% | 38.75% | 40.50% | 9.50 pp | -83 | 39 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 437 | 172 | 265 | 39.36% | 37.08% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 437 | 172 | 265 | 39.36% | 37.92% | 39.36% | 10.64 pp | -93 | 39 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 43 | 25 | 18 | 58.14% | 58.14% | 58.14% | 8.14 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 43 | 23 | 20 | 53.49% | 53.49% | 53.49% | 3.49 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 5 | -2.60 |

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
