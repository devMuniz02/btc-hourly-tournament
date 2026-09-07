# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T07:36:00.944936+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1273 | 985 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1149 | 784 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 856 | 546 | 309 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 858 | 600 | 256 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 190 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 190 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 190 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T16:00:00+00:00 | 191 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 546 | 276 | 270 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| Consolidated Market Hours | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 13 | -0.15 |
| BTC Market Hours | nn | NN | 546 | 269 | 277 | 49.27% | 51.25% | 50.83% | 0.73 pp | -8 | 51 | -0.16 |
| Consolidated Hourly | rf | RandomForest | 190 | 93 | 97 | 48.95% | 48.95% | 48.95% | 1.05 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 190 | 93 | 97 | 48.95% | 48.95% | 48.95% | 1.05 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | nn | NN | 600 | 286 | 314 | 47.67% | 47.50% | 47.92% | 2.33 pp | -28 | 51 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 774 | 373 | 401 | 48.19% | 46.67% | 47.71% | 1.81 pp | -28 | 45 | -0.62 |
| BTC Market Hours Daily | transformer | Transformer | 600 | 282 | 318 | 47.00% | 47.92% | 46.67% | 3.00 pp | -36 | 51 | -0.71 |
| BTC Market Hours | transformer | Transformer | 546 | 254 | 292 | 46.52% | 46.25% | 47.29% | 3.48 pp | -38 | 51 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 951 | 454 | 497 | 47.74% | 49.58% | 47.08% | 2.26 pp | -43 | 50 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 600 | 277 | 323 | 46.17% | 47.08% | 47.50% | 3.83 pp | -46 | 51 | -0.90 |
| Consolidated Market Hours Daily | lstm | LSTM | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 59 | 27 | 32 | 45.76% | 45.76% | 45.76% | 4.24 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 13 | -1.08 |
| BTC Daily | transformer | Transformer | 774 | 362 | 412 | 46.77% | 42.08% | 47.08% | 3.23 pp | -50 | 45 | -1.11 |
| BTC Market Hours | rf | RandomForest | 546 | 243 | 303 | 44.51% | 47.08% | 44.38% | 5.49 pp | -60 | 51 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 774 | 359 | 415 | 46.38% | 45.00% | 45.62% | 3.62 pp | -56 | 45 | -1.24 |
| BTC Hourly | transformer | Transformer | 951 | 443 | 508 | 46.58% | 45.00% | 44.38% | 3.42 pp | -65 | 50 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 600 | 266 | 334 | 44.33% | 46.25% | 43.75% | 5.67 pp | -68 | 51 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 190 | 85 | 105 | 44.74% | 44.74% | 44.74% | 5.26 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | nn | NN | 190 | 85 | 105 | 44.74% | 44.74% | 44.74% | 5.26 pp | -20 | 13 | -1.54 |
| Consolidated Market Hours | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 546 | 226 | 320 | 41.39% | 36.25% | 41.25% | 8.61 pp | -94 | 51 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 546 | 226 | 320 | 41.39% | 42.50% | 41.67% | 8.61 pp | -94 | 51 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 13 | -1.85 |
| BTC Market Hours Daily | xgb | XGBoost | 600 | 252 | 348 | 42.00% | 42.50% | 41.67% | 8.00 pp | -96 | 51 | -1.88 |
| BTC Hourly | rf | RandomForest | 951 | 423 | 528 | 44.48% | 44.17% | 43.54% | 5.52 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 951 | 421 | 530 | 44.27% | 42.08% | 42.71% | 5.73 pp | -109 | 50 | -2.18 |
| Consolidated Market Hours | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 600 | 237 | 363 | 39.50% | 35.83% | 38.96% | 10.50 pp | -126 | 51 | -2.47 |
| Consolidated Market Hours Daily | nn | NN | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 774 | 327 | 447 | 42.25% | 35.42% | 40.21% | 7.75 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 774 | 325 | 449 | 41.99% | 38.75% | 42.08% | 8.01 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 951 | 406 | 545 | 42.69% | 36.67% | 41.88% | 7.31 pp | -139 | 50 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 59 | 22 | 37 | 37.29% | 37.29% | 37.29% | 12.71 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 951 | 397 | 554 | 41.75% | 40.00% | 40.42% | 8.25 pp | -157 | 50 | -3.14 |
| BTC Daily | xgb | XGBoost | 784 | 306 | 478 | 39.03% | 35.00% | 36.25% | 10.97 pp | -172 | 45 | -3.82 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 951 | 454 | 497 | 47.74% | 49.58% | 47.08% | 2.26 pp | -43 | 50 | -0.86 |
| BTC Hourly | transformer | Transformer | 951 | 443 | 508 | 46.58% | 45.00% | 44.38% | 3.42 pp | -65 | 50 | -1.30 |
| BTC Hourly | rf | RandomForest | 951 | 423 | 528 | 44.48% | 44.17% | 43.54% | 5.52 pp | -105 | 50 | -2.10 |
| BTC Hourly | nn | NN | 951 | 421 | 530 | 44.27% | 42.08% | 42.71% | 5.73 pp | -109 | 50 | -2.18 |
| BTC Hourly | lstm | LSTM | 951 | 406 | 545 | 42.69% | 36.67% | 41.88% | 7.31 pp | -139 | 50 | -2.78 |
| BTC Hourly | xgb | XGBoost | 951 | 397 | 554 | 41.75% | 40.00% | 40.42% | 8.25 pp | -157 | 50 | -3.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 774 | 373 | 401 | 48.19% | 46.67% | 47.71% | 1.81 pp | -28 | 45 | -0.62 |
| BTC Daily | transformer | Transformer | 774 | 362 | 412 | 46.77% | 42.08% | 47.08% | 3.23 pp | -50 | 45 | -1.11 |
| BTC Daily | nn | NN | 774 | 359 | 415 | 46.38% | 45.00% | 45.62% | 3.62 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 774 | 327 | 447 | 42.25% | 35.42% | 40.21% | 7.75 pp | -120 | 45 | -2.67 |
| BTC Daily | rf | RandomForest | 774 | 325 | 449 | 41.99% | 38.75% | 42.08% | 8.01 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 784 | 306 | 478 | 39.03% | 35.00% | 36.25% | 10.97 pp | -172 | 45 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 546 | 276 | 270 | 50.55% | 47.50% | 50.21% | 0.55 pp | 6 | 51 | 0.12 |
| BTC Market Hours | nn | NN | 546 | 269 | 277 | 49.27% | 51.25% | 50.83% | 0.73 pp | -8 | 51 | -0.16 |
| BTC Market Hours | transformer | Transformer | 546 | 254 | 292 | 46.52% | 46.25% | 47.29% | 3.48 pp | -38 | 51 | -0.75 |
| BTC Market Hours | rf | RandomForest | 546 | 243 | 303 | 44.51% | 47.08% | 44.38% | 5.49 pp | -60 | 51 | -1.18 |
| BTC Market Hours | lstm | LSTM | 546 | 226 | 320 | 41.39% | 36.25% | 41.25% | 8.61 pp | -94 | 51 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 546 | 226 | 320 | 41.39% | 42.50% | 41.67% | 8.61 pp | -94 | 51 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 600 | 286 | 314 | 47.67% | 47.50% | 47.92% | 2.33 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 600 | 282 | 318 | 47.00% | 47.92% | 46.67% | 3.00 pp | -36 | 51 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 600 | 277 | 323 | 46.17% | 47.08% | 47.50% | 3.83 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 600 | 266 | 334 | 44.33% | 46.25% | 43.75% | 5.67 pp | -68 | 51 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 600 | 252 | 348 | 42.00% | 42.50% | 41.67% | 8.00 pp | -96 | 51 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 600 | 237 | 363 | 39.50% | 35.83% | 38.96% | 10.50 pp | -126 | 51 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 13 | -0.15 |
| Consolidated Hourly | rf | RandomForest | 190 | 93 | 97 | 48.95% | 48.95% | 48.95% | 1.05 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | nn | NN | 190 | 85 | 105 | 44.74% | 44.74% | 44.74% | 5.26 pp | -20 | 13 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 13 | -1.85 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 190 | 94 | 96 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 190 | 93 | 97 | 48.95% | 48.95% | 48.95% | 1.05 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 190 | 88 | 102 | 46.32% | 46.32% | 46.32% | 3.68 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | nn | NN | 190 | 85 | 105 | 44.74% | 44.74% | 44.74% | 5.26 pp | -20 | 13 | -1.54 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 13 | -1.85 |

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
