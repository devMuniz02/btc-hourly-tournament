# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T07:51:37.187332+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1192 | 904 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1068 | 703 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 710 | 465 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 712 | 519 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 115 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 115 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 18 | 97 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 15:00:00+00:00 | 115 | 18 | 97 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 465 | 225 | 240 | 48.39% | 44.17% | 48.39% | 1.61 pp | -15 | 45 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 693 | 338 | 355 | 48.77% | 45.42% | 49.17% | 1.23 pp | -17 | 42 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| BTC Market Hours | nn | NN | 465 | 218 | 247 | 46.88% | 47.92% | 46.88% | 3.12 pp | -29 | 45 | -0.64 |
| BTC Daily | transformer | Transformer | 693 | 332 | 361 | 47.91% | 45.83% | 49.17% | 2.09 pp | -29 | 42 | -0.69 |
| BTC Market Hours | transformer | Transformer | 465 | 216 | 249 | 46.45% | 40.42% | 46.45% | 3.55 pp | -33 | 45 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 519 | 238 | 281 | 45.86% | 46.25% | 46.25% | 4.14 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 870 | 412 | 458 | 47.36% | 47.50% | 47.71% | 2.64 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | nn | NN | 519 | 237 | 282 | 45.66% | 42.92% | 46.25% | 4.34 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 519 | 237 | 282 | 45.66% | 47.50% | 46.25% | 4.34 pp | -45 | 45 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Hourly | transformer | Transformer | 870 | 409 | 461 | 47.01% | 47.50% | 47.08% | 2.99 pp | -52 | 46 | -1.13 |
| BTC Daily | nn | NN | 693 | 322 | 371 | 46.46% | 42.50% | 48.54% | 3.54 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 465 | 200 | 265 | 43.01% | 43.33% | 43.01% | 6.99 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 465 | 198 | 267 | 42.58% | 40.42% | 42.58% | 7.42 pp | -69 | 45 | -1.53 |
| Consolidated Hourly | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |
| BTC Hourly | nn | NN | 870 | 391 | 479 | 44.94% | 45.83% | 43.96% | 5.06 pp | -88 | 46 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 465 | 189 | 276 | 40.65% | 39.58% | 40.65% | 9.35 pp | -87 | 45 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 519 | 215 | 304 | 41.43% | 41.67% | 41.46% | 8.57 pp | -89 | 45 | -1.98 |
| BTC Hourly | rf | RandomForest | 870 | 389 | 481 | 44.71% | 45.42% | 44.38% | 5.29 pp | -92 | 46 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| BTC Daily | lstm | LSTM | 693 | 301 | 392 | 43.43% | 38.33% | 42.50% | 6.57 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 519 | 208 | 311 | 40.08% | 38.33% | 40.83% | 9.92 pp | -103 | 45 | -2.29 |
| BTC Daily | rf | RandomForest | 693 | 297 | 396 | 42.86% | 40.42% | 43.33% | 7.14 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 519 | 206 | 313 | 39.69% | 37.08% | 39.38% | 10.31 pp | -107 | 45 | -2.38 |
| BTC Hourly | lstm | LSTM | 870 | 370 | 500 | 42.53% | 38.33% | 41.88% | 7.47 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 870 | 368 | 502 | 42.30% | 41.25% | 43.33% | 7.70 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 703 | 278 | 425 | 39.54% | 35.42% | 39.17% | 10.46 pp | -147 | 42 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 870 | 412 | 458 | 47.36% | 47.50% | 47.71% | 2.64 pp | -46 | 46 | -1.00 |
| BTC Hourly | transformer | Transformer | 870 | 409 | 461 | 47.01% | 47.50% | 47.08% | 2.99 pp | -52 | 46 | -1.13 |
| BTC Hourly | nn | NN | 870 | 391 | 479 | 44.94% | 45.83% | 43.96% | 5.06 pp | -88 | 46 | -1.91 |
| BTC Hourly | rf | RandomForest | 870 | 389 | 481 | 44.71% | 45.42% | 44.38% | 5.29 pp | -92 | 46 | -2.00 |
| BTC Hourly | lstm | LSTM | 870 | 370 | 500 | 42.53% | 38.33% | 41.88% | 7.47 pp | -130 | 46 | -2.83 |
| BTC Hourly | xgb | XGBoost | 870 | 368 | 502 | 42.30% | 41.25% | 43.33% | 7.70 pp | -134 | 46 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 693 | 338 | 355 | 48.77% | 45.42% | 49.17% | 1.23 pp | -17 | 42 | -0.40 |
| BTC Daily | transformer | Transformer | 693 | 332 | 361 | 47.91% | 45.83% | 49.17% | 2.09 pp | -29 | 42 | -0.69 |
| BTC Daily | nn | NN | 693 | 322 | 371 | 46.46% | 42.50% | 48.54% | 3.54 pp | -49 | 42 | -1.17 |
| BTC Daily | lstm | LSTM | 693 | 301 | 392 | 43.43% | 38.33% | 42.50% | 6.57 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 693 | 297 | 396 | 42.86% | 40.42% | 43.33% | 7.14 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 703 | 278 | 425 | 39.54% | 35.42% | 39.17% | 10.46 pp | -147 | 42 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 465 | 225 | 240 | 48.39% | 44.17% | 48.39% | 1.61 pp | -15 | 45 | -0.33 |
| BTC Market Hours | nn | NN | 465 | 218 | 247 | 46.88% | 47.92% | 46.88% | 3.12 pp | -29 | 45 | -0.64 |
| BTC Market Hours | transformer | Transformer | 465 | 216 | 249 | 46.45% | 40.42% | 46.45% | 3.55 pp | -33 | 45 | -0.73 |
| BTC Market Hours | rf | RandomForest | 465 | 200 | 265 | 43.01% | 43.33% | 43.01% | 6.99 pp | -65 | 45 | -1.44 |
| BTC Market Hours | lstm | LSTM | 465 | 198 | 267 | 42.58% | 40.42% | 42.58% | 7.42 pp | -69 | 45 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 465 | 189 | 276 | 40.65% | 39.58% | 40.65% | 9.35 pp | -87 | 45 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 519 | 238 | 281 | 45.86% | 46.25% | 46.25% | 4.14 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | nn | NN | 519 | 237 | 282 | 45.66% | 42.92% | 46.25% | 4.34 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 519 | 237 | 282 | 45.66% | 47.50% | 46.25% | 4.34 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 519 | 215 | 304 | 41.43% | 41.67% | 41.46% | 8.57 pp | -89 | 45 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 519 | 208 | 311 | 40.08% | 38.33% | 40.83% | 9.92 pp | -103 | 45 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 519 | 206 | 313 | 39.69% | 37.08% | 39.38% | 10.31 pp | -107 | 45 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 115 | 59 | 56 | 51.30% | 51.30% | 51.30% | 1.30 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 115 | 55 | 60 | 47.83% | 47.83% | 47.83% | 2.17 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 18 | 4 | 14 | 22.22% | 22.22% | 22.22% | 27.78 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
