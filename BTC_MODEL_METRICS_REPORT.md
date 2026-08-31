# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T10:01:26.741880+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1160 | 872 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1036 | 671 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 652 | 433 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 654 | 487 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 433 | 212 | 221 | 48.96% | 45.42% | 48.96% | 1.04 pp | -9 | 43 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 661 | 324 | 337 | 49.02% | 47.50% | 50.21% | 0.98 pp | -13 | 40 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Daily | transformer | Transformer | 661 | 319 | 342 | 48.26% | 45.42% | 49.58% | 1.74 pp | -23 | 40 | -0.57 |
| BTC Market Hours | nn | NN | 433 | 203 | 230 | 46.88% | 49.17% | 46.88% | 3.12 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 433 | 198 | 235 | 45.73% | 40.83% | 45.73% | 4.27 pp | -37 | 43 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 224 | 263 | 46.00% | 47.08% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Hourly | transformer | Transformer | 838 | 398 | 440 | 47.49% | 48.33% | 47.08% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Daily | nn | NN | 661 | 311 | 350 | 47.05% | 43.33% | 49.79% | 2.95 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | nn | NN | 487 | 221 | 266 | 45.38% | 43.33% | 45.62% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 221 | 266 | 45.38% | 45.42% | 45.42% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 838 | 392 | 446 | 46.78% | 43.33% | 46.46% | 3.22 pp | -54 | 45 | -1.20 |
| BTC Market Hours | lstm | LSTM | 433 | 187 | 246 | 43.19% | 43.33% | 43.19% | 6.81 pp | -59 | 43 | -1.37 |
| BTC Market Hours | rf | RandomForest | 433 | 186 | 247 | 42.96% | 42.92% | 42.96% | 7.04 pp | -61 | 43 | -1.42 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 838 | 380 | 458 | 45.35% | 44.58% | 45.00% | 4.65 pp | -78 | 45 | -1.73 |
| BTC Hourly | rf | RandomForest | 838 | 375 | 463 | 44.75% | 44.17% | 44.38% | 5.25 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 201 | 286 | 41.27% | 41.67% | 41.46% | 8.73 pp | -85 | 43 | -1.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 433 | 173 | 260 | 39.95% | 37.92% | 39.95% | 10.05 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 661 | 290 | 371 | 43.87% | 39.58% | 43.33% | 6.13 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 196 | 291 | 40.25% | 39.17% | 40.42% | 9.75 pp | -95 | 43 | -2.21 |
| BTC Daily | rf | RandomForest | 661 | 284 | 377 | 42.97% | 41.67% | 44.38% | 7.03 pp | -93 | 40 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 189 | 298 | 38.81% | 35.42% | 38.96% | 11.19 pp | -109 | 43 | -2.53 |
| BTC Hourly | lstm | LSTM | 838 | 361 | 477 | 43.08% | 40.42% | 42.71% | 6.92 pp | -116 | 45 | -2.58 |
| BTC Hourly | xgb | XGBoost | 838 | 355 | 483 | 42.36% | 40.00% | 42.71% | 7.64 pp | -128 | 45 | -2.84 |
| BTC Daily | xgb | XGBoost | 671 | 267 | 404 | 39.79% | 33.75% | 40.00% | 10.21 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 838 | 398 | 440 | 47.49% | 48.33% | 47.08% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 838 | 392 | 446 | 46.78% | 43.33% | 46.46% | 3.22 pp | -54 | 45 | -1.20 |
| BTC Hourly | nn | NN | 838 | 380 | 458 | 45.35% | 44.58% | 45.00% | 4.65 pp | -78 | 45 | -1.73 |
| BTC Hourly | rf | RandomForest | 838 | 375 | 463 | 44.75% | 44.17% | 44.38% | 5.25 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 838 | 361 | 477 | 43.08% | 40.42% | 42.71% | 6.92 pp | -116 | 45 | -2.58 |
| BTC Hourly | xgb | XGBoost | 838 | 355 | 483 | 42.36% | 40.00% | 42.71% | 7.64 pp | -128 | 45 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 661 | 324 | 337 | 49.02% | 47.50% | 50.21% | 0.98 pp | -13 | 40 | -0.33 |
| BTC Daily | transformer | Transformer | 661 | 319 | 342 | 48.26% | 45.42% | 49.58% | 1.74 pp | -23 | 40 | -0.57 |
| BTC Daily | nn | NN | 661 | 311 | 350 | 47.05% | 43.33% | 49.79% | 2.95 pp | -39 | 40 | -0.97 |
| BTC Daily | lstm | LSTM | 661 | 290 | 371 | 43.87% | 39.58% | 43.33% | 6.13 pp | -81 | 40 | -2.02 |
| BTC Daily | rf | RandomForest | 661 | 284 | 377 | 42.97% | 41.67% | 44.38% | 7.03 pp | -93 | 40 | -2.33 |
| BTC Daily | xgb | XGBoost | 671 | 267 | 404 | 39.79% | 33.75% | 40.00% | 10.21 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 433 | 212 | 221 | 48.96% | 45.42% | 48.96% | 1.04 pp | -9 | 43 | -0.21 |
| BTC Market Hours | nn | NN | 433 | 203 | 230 | 46.88% | 49.17% | 46.88% | 3.12 pp | -27 | 43 | -0.63 |
| BTC Market Hours | transformer | Transformer | 433 | 198 | 235 | 45.73% | 40.83% | 45.73% | 4.27 pp | -37 | 43 | -0.86 |
| BTC Market Hours | lstm | LSTM | 433 | 187 | 246 | 43.19% | 43.33% | 43.19% | 6.81 pp | -59 | 43 | -1.37 |
| BTC Market Hours | rf | RandomForest | 433 | 186 | 247 | 42.96% | 42.92% | 42.96% | 7.04 pp | -61 | 43 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 433 | 173 | 260 | 39.95% | 37.92% | 39.95% | 10.05 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 224 | 263 | 46.00% | 47.08% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | nn | NN | 487 | 221 | 266 | 45.38% | 43.33% | 45.62% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 221 | 266 | 45.38% | 45.42% | 45.42% | 4.62 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 201 | 286 | 41.27% | 41.67% | 41.46% | 8.73 pp | -85 | 43 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 196 | 291 | 40.25% | 39.17% | 40.42% | 9.75 pp | -95 | 43 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 189 | 298 | 38.81% | 35.42% | 38.96% | 11.19 pp | -109 | 43 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
