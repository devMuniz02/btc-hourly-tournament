# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T11:33:13.870032+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1275 | 987 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1151 | 786 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 858 | 548 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 860 | 602 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 193 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 193 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 60 | 133 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 60 | 133 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 548 | 265 | 283 | 48.36% | 45.42% | 47.50% | 1.64 pp | -18 | 52 | -0.35 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours | nn | NN | 548 | 261 | 287 | 47.63% | 50.83% | 49.38% | 2.37 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 548 | 260 | 288 | 47.45% | 47.92% | 47.92% | 2.55 pp | -28 | 52 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 776 | 373 | 403 | 48.07% | 45.83% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 602 | 284 | 318 | 47.18% | 50.00% | 48.33% | 2.82 pp | -34 | 51 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 953 | 454 | 499 | 47.64% | 49.58% | 47.08% | 2.36 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | nn | NN | 602 | 278 | 324 | 46.18% | 46.25% | 47.50% | 3.82 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 602 | 277 | 325 | 46.01% | 48.33% | 46.67% | 3.99 pp | -48 | 51 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| BTC Daily | transformer | Transformer | 776 | 362 | 414 | 46.65% | 42.08% | 46.88% | 3.35 pp | -52 | 45 | -1.16 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 776 | 359 | 417 | 46.26% | 44.17% | 45.21% | 3.74 pp | -58 | 45 | -1.29 |
| BTC Hourly | transformer | Transformer | 953 | 444 | 509 | 46.59% | 44.58% | 44.38% | 3.41 pp | -65 | 50 | -1.30 |
| BTC Market Hours | rf | RandomForest | 548 | 238 | 310 | 43.43% | 45.42% | 43.54% | 6.57 pp | -72 | 52 | -1.38 |
| BTC Market Hours | lstm | LSTM | 548 | 234 | 314 | 42.70% | 40.83% | 43.12% | 7.30 pp | -80 | 52 | -1.54 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 548 | 229 | 319 | 41.79% | 43.33% | 41.88% | 8.21 pp | -90 | 52 | -1.73 |
| Consolidated Hourly | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | rf | RandomForest | 602 | 252 | 350 | 41.86% | 44.17% | 41.25% | 8.14 pp | -98 | 51 | -1.92 |
| Consolidated Hourly | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 953 | 423 | 530 | 44.39% | 43.33% | 43.33% | 5.61 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 953 | 422 | 531 | 44.28% | 42.50% | 42.71% | 5.72 pp | -109 | 50 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 602 | 242 | 360 | 40.20% | 39.17% | 40.00% | 9.80 pp | -118 | 51 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 602 | 241 | 361 | 40.03% | 40.83% | 39.38% | 9.97 pp | -120 | 51 | -2.35 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 776 | 327 | 449 | 42.14% | 35.42% | 39.79% | 7.86 pp | -122 | 45 | -2.71 |
| BTC Hourly | lstm | LSTM | 953 | 408 | 545 | 42.81% | 37.50% | 42.29% | 7.19 pp | -137 | 50 | -2.74 |
| BTC Daily | rf | RandomForest | 776 | 325 | 451 | 41.88% | 38.75% | 41.88% | 8.12 pp | -126 | 45 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Hourly | xgb | XGBoost | 953 | 397 | 556 | 41.66% | 39.17% | 40.21% | 8.34 pp | -159 | 50 | -3.18 |
| BTC Daily | xgb | XGBoost | 786 | 307 | 479 | 39.06% | 35.42% | 36.25% | 10.94 pp | -172 | 45 | -3.82 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 953 | 454 | 499 | 47.64% | 49.58% | 47.08% | 2.36 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 953 | 444 | 509 | 46.59% | 44.58% | 44.38% | 3.41 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 953 | 423 | 530 | 44.39% | 43.33% | 43.33% | 5.61 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 953 | 422 | 531 | 44.28% | 42.50% | 42.71% | 5.72 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 953 | 408 | 545 | 42.81% | 37.50% | 42.29% | 7.19 pp | -137 | 50 | -2.74 |
| BTC Hourly | xgb | XGBoost | 953 | 397 | 556 | 41.66% | 39.17% | 40.21% | 8.34 pp | -159 | 50 | -3.18 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 776 | 373 | 403 | 48.07% | 45.83% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Daily | transformer | Transformer | 776 | 362 | 414 | 46.65% | 42.08% | 46.88% | 3.35 pp | -52 | 45 | -1.16 |
| BTC Daily | nn | NN | 776 | 359 | 417 | 46.26% | 44.17% | 45.21% | 3.74 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 776 | 327 | 449 | 42.14% | 35.42% | 39.79% | 7.86 pp | -122 | 45 | -2.71 |
| BTC Daily | rf | RandomForest | 776 | 325 | 451 | 41.88% | 38.75% | 41.88% | 8.12 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 786 | 307 | 479 | 39.06% | 35.42% | 36.25% | 10.94 pp | -172 | 45 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 548 | 265 | 283 | 48.36% | 45.42% | 47.50% | 1.64 pp | -18 | 52 | -0.35 |
| BTC Market Hours | nn | NN | 548 | 261 | 287 | 47.63% | 50.83% | 49.38% | 2.37 pp | -26 | 52 | -0.50 |
| BTC Market Hours | transformer | Transformer | 548 | 260 | 288 | 47.45% | 47.92% | 47.92% | 2.55 pp | -28 | 52 | -0.54 |
| BTC Market Hours | rf | RandomForest | 548 | 238 | 310 | 43.43% | 45.42% | 43.54% | 6.57 pp | -72 | 52 | -1.38 |
| BTC Market Hours | lstm | LSTM | 548 | 234 | 314 | 42.70% | 40.83% | 43.12% | 7.30 pp | -80 | 52 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 548 | 229 | 319 | 41.79% | 43.33% | 41.88% | 8.21 pp | -90 | 52 | -1.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 602 | 284 | 318 | 47.18% | 50.00% | 48.33% | 2.82 pp | -34 | 51 | -0.67 |
| BTC Market Hours Daily | nn | NN | 602 | 278 | 324 | 46.18% | 46.25% | 47.50% | 3.82 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 602 | 277 | 325 | 46.01% | 48.33% | 46.67% | 3.99 pp | -48 | 51 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 602 | 252 | 350 | 41.86% | 44.17% | 41.25% | 8.14 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 602 | 242 | 360 | 40.20% | 39.17% | 40.00% | 9.80 pp | -118 | 51 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 602 | 241 | 361 | 40.03% | 40.83% | 39.38% | 9.97 pp | -120 | 51 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
