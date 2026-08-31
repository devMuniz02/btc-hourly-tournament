# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T06:38:38.345913+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1158 | 870 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1034 | 669 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 650 | 431 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 652 | 485 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T00:00:00+00:00 | 83 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T00:00:00+00:00 | 83 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T00:00:00+00:00 | 83 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T00:00:00+00:00 | 84 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 9 | 0.78 |
| BTC Daily | mlp_sklearn | MLPClassifier | 659 | 343 | 316 | 52.05% | 50.00% | 53.12% | 2.05 pp | 27 | 40 | 0.68 |
| Consolidated Hourly | rf | RandomForest | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | xgb | XGBoost | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 431 | 219 | 212 | 50.81% | 47.50% | 50.81% | 0.81 pp | 7 | 43 | 0.16 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 9 | -0.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 836 | 415 | 421 | 49.64% | 48.33% | 49.79% | 0.36 pp | -6 | 45 | -0.13 |
| BTC Market Hours | nn | NN | 431 | 212 | 219 | 49.19% | 50.42% | 49.19% | 0.81 pp | -7 | 43 | -0.16 |
| BTC Daily | nn | NN | 659 | 318 | 341 | 48.25% | 46.67% | 50.21% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Market Hours Daily | nn | NN | 485 | 228 | 257 | 47.01% | 45.00% | 47.29% | 2.99 pp | -29 | 42 | -0.69 |
| Consolidated Hourly | nn | NN | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| BTC Daily | transformer | Transformer | 659 | 312 | 347 | 47.34% | 44.58% | 49.17% | 2.66 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 485 | 223 | 262 | 45.98% | 47.50% | 46.25% | 4.02 pp | -39 | 42 | -0.93 |
| BTC Hourly | nn | NN | 836 | 397 | 439 | 47.49% | 47.08% | 46.88% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | transformer | Transformer | 836 | 397 | 439 | 47.49% | 46.67% | 45.21% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Market Hours | transformer | Transformer | 431 | 195 | 236 | 45.24% | 40.83% | 45.24% | 4.76 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 485 | 222 | 263 | 45.77% | 43.33% | 45.83% | 4.23 pp | -41 | 42 | -0.98 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 836 | 393 | 443 | 47.01% | 46.25% | 46.04% | 2.99 pp | -50 | 45 | -1.11 |
| BTC Market Hours | rf | RandomForest | 431 | 189 | 242 | 43.85% | 44.17% | 43.85% | 6.15 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 659 | 303 | 356 | 45.98% | 40.00% | 45.83% | 4.02 pp | -53 | 40 | -1.32 |
| BTC Market Hours Daily | rf | RandomForest | 485 | 214 | 271 | 44.12% | 43.75% | 44.17% | 5.88 pp | -57 | 42 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 431 | 184 | 247 | 42.69% | 40.83% | 42.69% | 7.31 pp | -63 | 43 | -1.47 |
| BTC Daily | rf | RandomForest | 659 | 298 | 361 | 45.22% | 42.50% | 46.25% | 4.78 pp | -63 | 40 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 431 | 174 | 257 | 40.37% | 38.33% | 40.37% | 9.63 pp | -83 | 43 | -1.93 |
| BTC Hourly | lstm | LSTM | 836 | 374 | 462 | 44.74% | 43.75% | 45.21% | 5.26 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 485 | 201 | 284 | 41.44% | 39.58% | 41.46% | 8.56 pp | -83 | 42 | -1.98 |
| BTC Hourly | xgb | XGBoost | 836 | 368 | 468 | 44.02% | 43.33% | 43.96% | 5.98 pp | -100 | 45 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 485 | 194 | 291 | 40.00% | 36.67% | 40.00% | 10.00 pp | -97 | 42 | -2.31 |
| BTC Daily | xgb | XGBoost | 669 | 270 | 399 | 40.36% | 34.58% | 41.04% | 9.64 pp | -129 | 40 | -3.23 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 836 | 415 | 421 | 49.64% | 48.33% | 49.79% | 0.36 pp | -6 | 45 | -0.13 |
| BTC Hourly | nn | NN | 836 | 397 | 439 | 47.49% | 47.08% | 46.88% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | transformer | Transformer | 836 | 397 | 439 | 47.49% | 46.67% | 45.21% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | rf | RandomForest | 836 | 393 | 443 | 47.01% | 46.25% | 46.04% | 2.99 pp | -50 | 45 | -1.11 |
| BTC Hourly | lstm | LSTM | 836 | 374 | 462 | 44.74% | 43.75% | 45.21% | 5.26 pp | -88 | 45 | -1.96 |
| BTC Hourly | xgb | XGBoost | 836 | 368 | 468 | 44.02% | 43.33% | 43.96% | 5.98 pp | -100 | 45 | -2.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 659 | 343 | 316 | 52.05% | 50.00% | 53.12% | 2.05 pp | 27 | 40 | 0.68 |
| BTC Daily | nn | NN | 659 | 318 | 341 | 48.25% | 46.67% | 50.21% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Daily | transformer | Transformer | 659 | 312 | 347 | 47.34% | 44.58% | 49.17% | 2.66 pp | -35 | 40 | -0.88 |
| BTC Daily | lstm | LSTM | 659 | 303 | 356 | 45.98% | 40.00% | 45.83% | 4.02 pp | -53 | 40 | -1.32 |
| BTC Daily | rf | RandomForest | 659 | 298 | 361 | 45.22% | 42.50% | 46.25% | 4.78 pp | -63 | 40 | -1.57 |
| BTC Daily | xgb | XGBoost | 669 | 270 | 399 | 40.36% | 34.58% | 41.04% | 9.64 pp | -129 | 40 | -3.23 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 431 | 219 | 212 | 50.81% | 47.50% | 50.81% | 0.81 pp | 7 | 43 | 0.16 |
| BTC Market Hours | nn | NN | 431 | 212 | 219 | 49.19% | 50.42% | 49.19% | 0.81 pp | -7 | 43 | -0.16 |
| BTC Market Hours | transformer | Transformer | 431 | 195 | 236 | 45.24% | 40.83% | 45.24% | 4.76 pp | -41 | 43 | -0.95 |
| BTC Market Hours | rf | RandomForest | 431 | 189 | 242 | 43.85% | 44.17% | 43.85% | 6.15 pp | -53 | 43 | -1.23 |
| BTC Market Hours | lstm | LSTM | 431 | 184 | 247 | 42.69% | 40.83% | 42.69% | 7.31 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 431 | 174 | 257 | 40.37% | 38.33% | 40.37% | 9.63 pp | -83 | 43 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 485 | 228 | 257 | 47.01% | 45.00% | 47.29% | 2.99 pp | -29 | 42 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 485 | 223 | 262 | 45.98% | 47.50% | 46.25% | 4.02 pp | -39 | 42 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 485 | 222 | 263 | 45.77% | 43.33% | 45.83% | 4.23 pp | -41 | 42 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 485 | 214 | 271 | 44.12% | 43.75% | 44.17% | 5.88 pp | -57 | 42 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 485 | 201 | 284 | 41.44% | 39.58% | 41.46% | 8.56 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 485 | 194 | 291 | 40.00% | 36.67% | 40.00% | 10.00 pp | -97 | 42 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | rf | RandomForest | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | xgb | XGBoost | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | nn | NN | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 45 | 38 | 54.22% | 54.22% | 54.22% | 4.22 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 35 | 48 | 42.17% | 42.17% | 42.17% | 7.83 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
