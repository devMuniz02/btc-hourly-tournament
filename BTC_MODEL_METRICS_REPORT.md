# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T10:10:48.584858+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1150 | 785 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 857 | 547 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 859 | 601 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 191 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 191 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 59 | 132 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 17:00:00+00:00 | 191 | 59 | 132 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 547 | 276 | 271 | 50.46% | 47.08% | 50.00% | 0.46 pp | 5 | 51 | 0.10 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| BTC Market Hours | nn | NN | 547 | 270 | 277 | 49.36% | 51.67% | 50.83% | 0.64 pp | -7 | 51 | -0.14 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 601 | 287 | 314 | 47.75% | 47.92% | 48.12% | 2.25 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 601 | 283 | 318 | 47.09% | 47.92% | 46.88% | 2.91 pp | -35 | 51 | -0.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 775 | 372 | 403 | 48.00% | 45.83% | 47.29% | 2.00 pp | -31 | 45 | -0.69 |
| BTC Market Hours | transformer | Transformer | 547 | 254 | 293 | 46.44% | 45.83% | 47.08% | 3.56 pp | -39 | 51 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 601 | 278 | 323 | 46.26% | 47.08% | 47.50% | 3.74 pp | -45 | 51 | -0.88 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 953 | 454 | 499 | 47.64% | 49.58% | 47.08% | 2.36 pp | -45 | 50 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| BTC Daily | transformer | Transformer | 775 | 361 | 414 | 46.58% | 41.67% | 46.67% | 3.42 pp | -53 | 45 | -1.18 |
| BTC Market Hours | rf | RandomForest | 547 | 243 | 304 | 44.42% | 46.67% | 44.17% | 5.58 pp | -61 | 51 | -1.20 |
| BTC Daily | nn | NN | 775 | 358 | 417 | 46.19% | 44.17% | 45.21% | 3.81 pp | -59 | 45 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 601 | 267 | 334 | 44.43% | 46.25% | 43.75% | 5.57 pp | -67 | 51 | -1.31 |
| BTC Hourly | transformer | Transformer | 953 | 443 | 510 | 46.48% | 44.17% | 44.17% | 3.52 pp | -67 | 50 | -1.34 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| BTC Market Hours | lstm | LSTM | 547 | 226 | 321 | 41.32% | 35.83% | 41.04% | 8.68 pp | -95 | 51 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 547 | 226 | 321 | 41.32% | 42.50% | 41.46% | 8.68 pp | -95 | 51 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 601 | 252 | 349 | 41.93% | 42.08% | 41.46% | 8.07 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |
| BTC Hourly | rf | RandomForest | 953 | 423 | 530 | 44.39% | 43.33% | 43.33% | 5.61 pp | -107 | 50 | -2.14 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Hourly | nn | NN | 953 | 421 | 532 | 44.18% | 42.08% | 42.50% | 5.82 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 601 | 237 | 364 | 39.43% | 35.42% | 38.96% | 10.57 pp | -127 | 51 | -2.49 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 775 | 327 | 448 | 42.19% | 35.42% | 40.00% | 7.81 pp | -121 | 45 | -2.69 |
| BTC Hourly | lstm | LSTM | 953 | 407 | 546 | 42.71% | 37.08% | 42.08% | 7.29 pp | -139 | 50 | -2.78 |
| BTC Daily | rf | RandomForest | 775 | 324 | 451 | 41.81% | 38.33% | 41.67% | 8.19 pp | -127 | 45 | -2.82 |
| BTC Hourly | xgb | XGBoost | 953 | 397 | 556 | 41.66% | 39.17% | 40.21% | 8.34 pp | -159 | 50 | -3.18 |
| BTC Daily | xgb | XGBoost | 785 | 306 | 479 | 38.98% | 35.00% | 36.04% | 11.02 pp | -173 | 45 | -3.84 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 953 | 454 | 499 | 47.64% | 49.58% | 47.08% | 2.36 pp | -45 | 50 | -0.90 |
| BTC Hourly | transformer | Transformer | 953 | 443 | 510 | 46.48% | 44.17% | 44.17% | 3.52 pp | -67 | 50 | -1.34 |
| BTC Hourly | rf | RandomForest | 953 | 423 | 530 | 44.39% | 43.33% | 43.33% | 5.61 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 953 | 421 | 532 | 44.18% | 42.08% | 42.50% | 5.82 pp | -111 | 50 | -2.22 |
| BTC Hourly | lstm | LSTM | 953 | 407 | 546 | 42.71% | 37.08% | 42.08% | 7.29 pp | -139 | 50 | -2.78 |
| BTC Hourly | xgb | XGBoost | 953 | 397 | 556 | 41.66% | 39.17% | 40.21% | 8.34 pp | -159 | 50 | -3.18 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 775 | 372 | 403 | 48.00% | 45.83% | 47.29% | 2.00 pp | -31 | 45 | -0.69 |
| BTC Daily | transformer | Transformer | 775 | 361 | 414 | 46.58% | 41.67% | 46.67% | 3.42 pp | -53 | 45 | -1.18 |
| BTC Daily | nn | NN | 775 | 358 | 417 | 46.19% | 44.17% | 45.21% | 3.81 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 775 | 327 | 448 | 42.19% | 35.42% | 40.00% | 7.81 pp | -121 | 45 | -2.69 |
| BTC Daily | rf | RandomForest | 775 | 324 | 451 | 41.81% | 38.33% | 41.67% | 8.19 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 785 | 306 | 479 | 38.98% | 35.00% | 36.04% | 11.02 pp | -173 | 45 | -3.84 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 547 | 276 | 271 | 50.46% | 47.08% | 50.00% | 0.46 pp | 5 | 51 | 0.10 |
| BTC Market Hours | nn | NN | 547 | 270 | 277 | 49.36% | 51.67% | 50.83% | 0.64 pp | -7 | 51 | -0.14 |
| BTC Market Hours | transformer | Transformer | 547 | 254 | 293 | 46.44% | 45.83% | 47.08% | 3.56 pp | -39 | 51 | -0.76 |
| BTC Market Hours | rf | RandomForest | 547 | 243 | 304 | 44.42% | 46.67% | 44.17% | 5.58 pp | -61 | 51 | -1.20 |
| BTC Market Hours | lstm | LSTM | 547 | 226 | 321 | 41.32% | 35.83% | 41.04% | 8.68 pp | -95 | 51 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 547 | 226 | 321 | 41.32% | 42.50% | 41.46% | 8.68 pp | -95 | 51 | -1.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 601 | 287 | 314 | 47.75% | 47.92% | 48.12% | 2.25 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 601 | 283 | 318 | 47.09% | 47.92% | 46.88% | 2.91 pp | -35 | 51 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 601 | 278 | 323 | 46.26% | 47.08% | 47.50% | 3.74 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 601 | 267 | 334 | 44.43% | 46.25% | 43.75% | 5.57 pp | -67 | 51 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 601 | 252 | 349 | 41.93% | 42.08% | 41.46% | 8.07 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 601 | 237 | 364 | 39.43% | 35.42% | 38.96% | 10.57 pp | -127 | 51 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
