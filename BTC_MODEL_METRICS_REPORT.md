# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T23:18:30.116881+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1283 | 995 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1159 | 794 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 877 | 556 | 320 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 22:00:00+00:00 | 879 | 610 | 267 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 201 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 22:00:00+00:00 | 201 | 64 | 137 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 556 | 272 | 284 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 556 | 265 | 291 | 47.66% | 51.67% | 49.38% | 2.34 pp | -26 | 52 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| BTC Market Hours | transformer | Transformer | 556 | 263 | 293 | 47.30% | 47.08% | 47.71% | 2.70 pp | -30 | 52 | -0.58 |
| BTC Daily | mlp_sklearn | MLPClassifier | 784 | 378 | 406 | 48.21% | 46.25% | 47.92% | 1.79 pp | -28 | 45 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 610 | 286 | 324 | 46.89% | 50.00% | 47.71% | 3.11 pp | -38 | 52 | -0.73 |
| BTC Market Hours Daily | nn | NN | 610 | 285 | 325 | 46.72% | 47.92% | 48.33% | 3.28 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 610 | 284 | 326 | 46.56% | 49.58% | 47.29% | 3.44 pp | -42 | 52 | -0.81 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 961 | 458 | 503 | 47.66% | 49.58% | 47.08% | 2.34 pp | -45 | 50 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 784 | 364 | 420 | 46.43% | 40.42% | 46.46% | 3.57 pp | -56 | 45 | -1.24 |
| BTC Daily | nn | NN | 784 | 363 | 421 | 46.30% | 44.58% | 45.21% | 3.70 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 961 | 447 | 514 | 46.51% | 44.58% | 43.75% | 3.49 pp | -67 | 50 | -1.34 |
| BTC Market Hours | rf | RandomForest | 556 | 241 | 315 | 43.35% | 45.83% | 43.33% | 6.65 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 556 | 240 | 316 | 43.17% | 41.67% | 43.54% | 6.83 pp | -76 | 52 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 556 | 235 | 321 | 42.27% | 45.00% | 42.50% | 7.73 pp | -86 | 52 | -1.65 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 610 | 256 | 354 | 41.97% | 44.17% | 41.25% | 8.03 pp | -98 | 52 | -1.88 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 610 | 248 | 362 | 40.66% | 40.42% | 40.62% | 9.34 pp | -114 | 52 | -2.19 |
| BTC Hourly | nn | NN | 961 | 425 | 536 | 44.22% | 42.08% | 42.50% | 5.78 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 961 | 425 | 536 | 44.22% | 42.50% | 42.92% | 5.78 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 610 | 247 | 363 | 40.49% | 42.08% | 39.79% | 9.51 pp | -116 | 52 | -2.23 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 784 | 329 | 455 | 41.96% | 33.75% | 39.79% | 8.04 pp | -126 | 45 | -2.80 |
| BTC Hourly | lstm | LSTM | 961 | 410 | 551 | 42.66% | 37.08% | 41.88% | 7.34 pp | -141 | 50 | -2.82 |
| BTC Daily | rf | RandomForest | 784 | 328 | 456 | 41.84% | 37.92% | 41.67% | 8.16 pp | -128 | 45 | -2.84 |
| BTC Hourly | xgb | XGBoost | 961 | 397 | 564 | 41.31% | 37.08% | 39.17% | 8.69 pp | -167 | 50 | -3.34 |
| BTC Daily | xgb | XGBoost | 794 | 310 | 484 | 39.04% | 35.42% | 36.67% | 10.96 pp | -174 | 45 | -3.87 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 961 | 458 | 503 | 47.66% | 49.58% | 47.08% | 2.34 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 961 | 447 | 514 | 46.51% | 44.58% | 43.75% | 3.49 pp | -67 | 50 | -1.34 |
| BTC Hourly | nn | NN | 961 | 425 | 536 | 44.22% | 42.08% | 42.50% | 5.78 pp | -111 | 50 | -2.22 |
| BTC Hourly | rf | RandomForest | 961 | 425 | 536 | 44.22% | 42.50% | 42.92% | 5.78 pp | -111 | 50 | -2.22 |
| BTC Hourly | lstm | LSTM | 961 | 410 | 551 | 42.66% | 37.08% | 41.88% | 7.34 pp | -141 | 50 | -2.82 |
| BTC Hourly | xgb | XGBoost | 961 | 397 | 564 | 41.31% | 37.08% | 39.17% | 8.69 pp | -167 | 50 | -3.34 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 784 | 378 | 406 | 48.21% | 46.25% | 47.92% | 1.79 pp | -28 | 45 | -0.62 |
| BTC Daily | transformer | Transformer | 784 | 364 | 420 | 46.43% | 40.42% | 46.46% | 3.57 pp | -56 | 45 | -1.24 |
| BTC Daily | nn | NN | 784 | 363 | 421 | 46.30% | 44.58% | 45.21% | 3.70 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 784 | 329 | 455 | 41.96% | 33.75% | 39.79% | 8.04 pp | -126 | 45 | -2.80 |
| BTC Daily | rf | RandomForest | 784 | 328 | 456 | 41.84% | 37.92% | 41.67% | 8.16 pp | -128 | 45 | -2.84 |
| BTC Daily | xgb | XGBoost | 794 | 310 | 484 | 39.04% | 35.42% | 36.67% | 10.96 pp | -174 | 45 | -3.87 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 556 | 272 | 284 | 48.92% | 47.92% | 47.92% | 1.08 pp | -12 | 52 | -0.23 |
| BTC Market Hours | nn | NN | 556 | 265 | 291 | 47.66% | 51.67% | 49.38% | 2.34 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 556 | 263 | 293 | 47.30% | 47.08% | 47.71% | 2.70 pp | -30 | 52 | -0.58 |
| BTC Market Hours | rf | RandomForest | 556 | 241 | 315 | 43.35% | 45.83% | 43.33% | 6.65 pp | -74 | 52 | -1.42 |
| BTC Market Hours | lstm | LSTM | 556 | 240 | 316 | 43.17% | 41.67% | 43.54% | 6.83 pp | -76 | 52 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 556 | 235 | 321 | 42.27% | 45.00% | 42.50% | 7.73 pp | -86 | 52 | -1.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 610 | 286 | 324 | 46.89% | 50.00% | 47.71% | 3.11 pp | -38 | 52 | -0.73 |
| BTC Market Hours Daily | nn | NN | 610 | 285 | 325 | 46.72% | 47.92% | 48.33% | 3.28 pp | -40 | 52 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 610 | 284 | 326 | 46.56% | 49.58% | 47.29% | 3.44 pp | -42 | 52 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 610 | 256 | 354 | 41.97% | 44.17% | 41.25% | 8.03 pp | -98 | 52 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 610 | 248 | 362 | 40.66% | 40.42% | 40.62% | 9.34 pp | -114 | 52 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 610 | 247 | 363 | 40.49% | 42.08% | 39.79% | 9.51 pp | -116 | 52 | -2.23 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 201 | 93 | 108 | 46.27% | 46.27% | 46.27% | 3.73 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 201 | 89 | 112 | 44.28% | 44.28% | 44.28% | 5.72 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 201 | 87 | 114 | 43.28% | 43.28% | 43.28% | 6.72 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 31 | 33 | 48.44% | 48.44% | 48.44% | 1.56 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 29 | 35 | 45.31% | 45.31% | 45.31% | 4.69 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
