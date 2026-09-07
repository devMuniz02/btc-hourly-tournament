# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T06:40:15.052630+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1272 | 984 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1148 | 783 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 855 | 545 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 857 | 599 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 189 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 189 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 189 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 190 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 545 | 275 | 270 | 50.46% | 47.08% | 50.00% | 0.46 pp | 5 | 51 | 0.10 |
| Consolidated Market Hours | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| BTC Market Hours | nn | NN | 545 | 269 | 276 | 49.36% | 51.67% | 50.83% | 0.64 pp | -7 | 51 | -0.14 |
| Consolidated Hourly | rf | RandomForest | 189 | 93 | 96 | 49.21% | 49.21% | 49.21% | 0.79 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 189 | 93 | 96 | 49.21% | 49.21% | 49.21% | 0.79 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | nn | NN | 599 | 286 | 313 | 47.75% | 47.50% | 48.12% | 2.25 pp | -27 | 51 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 773 | 373 | 400 | 48.25% | 46.67% | 47.92% | 1.75 pp | -27 | 45 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 599 | 282 | 317 | 47.08% | 48.33% | 46.67% | 2.92 pp | -35 | 51 | -0.69 |
| BTC Market Hours | transformer | Transformer | 545 | 253 | 292 | 46.42% | 46.25% | 47.29% | 3.58 pp | -39 | 51 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 950 | 454 | 496 | 47.79% | 50.00% | 47.08% | 2.21 pp | -42 | 50 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 599 | 277 | 322 | 46.24% | 47.50% | 47.71% | 3.76 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 773 | 361 | 412 | 46.70% | 41.67% | 47.08% | 3.30 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 545 | 242 | 303 | 44.40% | 46.67% | 44.17% | 5.60 pp | -61 | 51 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 773 | 358 | 415 | 46.31% | 44.58% | 45.62% | 3.69 pp | -57 | 45 | -1.27 |
| BTC Hourly | transformer | Transformer | 950 | 443 | 507 | 46.63% | 45.42% | 44.38% | 3.37 pp | -64 | 50 | -1.28 |
| Consolidated Hourly | lstm | LSTM | 189 | 86 | 103 | 45.50% | 45.50% | 45.50% | 4.50 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 189 | 86 | 103 | 45.50% | 45.50% | 45.50% | 4.50 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 599 | 265 | 334 | 44.24% | 46.25% | 43.54% | 5.76 pp | -69 | 51 | -1.35 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 189 | 85 | 104 | 44.97% | 44.97% | 44.97% | 5.03 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 189 | 85 | 104 | 44.97% | 44.97% | 44.97% | 5.03 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 545 | 226 | 319 | 41.47% | 36.67% | 41.46% | 8.53 pp | -93 | 51 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 545 | 226 | 319 | 41.47% | 42.92% | 41.67% | 8.53 pp | -93 | 51 | -1.82 |
| BTC Market Hours Daily | xgb | XGBoost | 599 | 251 | 348 | 41.90% | 42.50% | 41.46% | 8.10 pp | -97 | 51 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 13 | -1.92 |
| BTC Hourly | rf | RandomForest | 950 | 423 | 527 | 44.53% | 44.58% | 43.75% | 5.47 pp | -104 | 50 | -2.08 |
| BTC Hourly | nn | NN | 950 | 421 | 529 | 44.32% | 42.50% | 42.92% | 5.68 pp | -108 | 50 | -2.16 |
| Consolidated Market Hours | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 599 | 237 | 362 | 39.57% | 35.83% | 38.96% | 10.43 pp | -125 | 51 | -2.45 |
| Consolidated Market Hours Daily | nn | NN | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 773 | 326 | 447 | 42.17% | 35.42% | 40.21% | 7.83 pp | -121 | 45 | -2.69 |
| BTC Hourly | lstm | LSTM | 950 | 406 | 544 | 42.74% | 36.67% | 42.08% | 7.26 pp | -138 | 50 | -2.76 |
| BTC Daily | rf | RandomForest | 773 | 324 | 449 | 41.91% | 38.33% | 42.08% | 8.09 pp | -125 | 45 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 950 | 397 | 553 | 41.79% | 40.00% | 40.42% | 8.21 pp | -156 | 50 | -3.12 |
| BTC Daily | xgb | XGBoost | 783 | 306 | 477 | 39.08% | 35.00% | 36.46% | 10.92 pp | -171 | 45 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 950 | 454 | 496 | 47.79% | 50.00% | 47.08% | 2.21 pp | -42 | 50 | -0.84 |
| BTC Hourly | transformer | Transformer | 950 | 443 | 507 | 46.63% | 45.42% | 44.38% | 3.37 pp | -64 | 50 | -1.28 |
| BTC Hourly | rf | RandomForest | 950 | 423 | 527 | 44.53% | 44.58% | 43.75% | 5.47 pp | -104 | 50 | -2.08 |
| BTC Hourly | nn | NN | 950 | 421 | 529 | 44.32% | 42.50% | 42.92% | 5.68 pp | -108 | 50 | -2.16 |
| BTC Hourly | lstm | LSTM | 950 | 406 | 544 | 42.74% | 36.67% | 42.08% | 7.26 pp | -138 | 50 | -2.76 |
| BTC Hourly | xgb | XGBoost | 950 | 397 | 553 | 41.79% | 40.00% | 40.42% | 8.21 pp | -156 | 50 | -3.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 773 | 373 | 400 | 48.25% | 46.67% | 47.92% | 1.75 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 773 | 361 | 412 | 46.70% | 41.67% | 47.08% | 3.30 pp | -51 | 45 | -1.13 |
| BTC Daily | nn | NN | 773 | 358 | 415 | 46.31% | 44.58% | 45.62% | 3.69 pp | -57 | 45 | -1.27 |
| BTC Daily | lstm | LSTM | 773 | 326 | 447 | 42.17% | 35.42% | 40.21% | 7.83 pp | -121 | 45 | -2.69 |
| BTC Daily | rf | RandomForest | 773 | 324 | 449 | 41.91% | 38.33% | 42.08% | 8.09 pp | -125 | 45 | -2.78 |
| BTC Daily | xgb | XGBoost | 783 | 306 | 477 | 39.08% | 35.00% | 36.46% | 10.92 pp | -171 | 45 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 545 | 275 | 270 | 50.46% | 47.08% | 50.00% | 0.46 pp | 5 | 51 | 0.10 |
| BTC Market Hours | nn | NN | 545 | 269 | 276 | 49.36% | 51.67% | 50.83% | 0.64 pp | -7 | 51 | -0.14 |
| BTC Market Hours | transformer | Transformer | 545 | 253 | 292 | 46.42% | 46.25% | 47.29% | 3.58 pp | -39 | 51 | -0.76 |
| BTC Market Hours | rf | RandomForest | 545 | 242 | 303 | 44.40% | 46.67% | 44.17% | 5.60 pp | -61 | 51 | -1.20 |
| BTC Market Hours | lstm | LSTM | 545 | 226 | 319 | 41.47% | 36.67% | 41.46% | 8.53 pp | -93 | 51 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 545 | 226 | 319 | 41.47% | 42.92% | 41.67% | 8.53 pp | -93 | 51 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 599 | 286 | 313 | 47.75% | 47.50% | 48.12% | 2.25 pp | -27 | 51 | -0.53 |
| BTC Market Hours Daily | transformer | Transformer | 599 | 282 | 317 | 47.08% | 48.33% | 46.67% | 2.92 pp | -35 | 51 | -0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 599 | 277 | 322 | 46.24% | 47.50% | 47.71% | 3.76 pp | -45 | 51 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 599 | 265 | 334 | 44.24% | 46.25% | 43.54% | 5.76 pp | -69 | 51 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 599 | 251 | 348 | 41.90% | 42.50% | 41.46% | 8.10 pp | -97 | 51 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 599 | 237 | 362 | 39.57% | 35.83% | 38.96% | 10.43 pp | -125 | 51 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 189 | 93 | 96 | 49.21% | 49.21% | 49.21% | 0.79 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | xgb | XGBoost | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 189 | 86 | 103 | 45.50% | 45.50% | 45.50% | 4.50 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 189 | 85 | 104 | 44.97% | 44.97% | 44.97% | 5.03 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 189 | 93 | 96 | 49.21% | 49.21% | 49.21% | 0.79 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 189 | 86 | 103 | 45.50% | 45.50% | 45.50% | 4.50 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 189 | 85 | 104 | 44.97% | 44.97% | 44.97% | 5.03 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 5 | 0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | nn | NN | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
