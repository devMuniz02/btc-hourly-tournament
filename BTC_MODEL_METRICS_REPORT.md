# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T23:25:53.430866+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1315 | 1027 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1191 | 826 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 22:00:00+00:00 | 935 | 588 | 346 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 22:00:00+00:00 | 937 | 642 | 293 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 229 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T12:00:00+00:00 | 230 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 588 | 285 | 303 | 48.47% | 47.92% | 47.50% | 1.53 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 588 | 284 | 304 | 48.30% | 52.50% | 50.00% | 1.70 pp | -20 | 55 | -0.36 |
| Consolidated Hourly | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| BTC Market Hours | transformer | Transformer | 588 | 277 | 311 | 47.11% | 47.08% | 46.46% | 2.89 pp | -34 | 55 | -0.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 816 | 392 | 424 | 48.04% | 45.42% | 46.88% | 1.96 pp | -32 | 47 | -0.68 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 642 | 301 | 341 | 46.88% | 49.17% | 47.29% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 642 | 301 | 341 | 46.88% | 48.33% | 48.12% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 642 | 301 | 341 | 46.88% | 49.17% | 47.71% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 993 | 471 | 522 | 47.43% | 49.17% | 46.25% | 2.57 pp | -51 | 51 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 816 | 381 | 435 | 46.69% | 45.42% | 45.42% | 3.31 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 816 | 379 | 437 | 46.45% | 40.00% | 46.25% | 3.55 pp | -58 | 47 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 993 | 461 | 532 | 46.42% | 44.58% | 44.17% | 3.58 pp | -71 | 51 | -1.39 |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 588 | 251 | 337 | 42.69% | 42.50% | 42.50% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 588 | 251 | 337 | 42.69% | 43.33% | 42.92% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 588 | 251 | 337 | 42.69% | 44.17% | 43.12% | 7.31 pp | -86 | 55 | -1.56 |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 34 | 47 | 41.98% | 41.98% | 41.98% | 8.02 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 642 | 266 | 376 | 41.43% | 42.08% | 40.83% | 8.57 pp | -110 | 55 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 642 | 264 | 378 | 41.12% | 42.92% | 40.42% | 8.88 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 642 | 263 | 379 | 40.97% | 42.08% | 40.83% | 9.03 pp | -116 | 55 | -2.11 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| BTC Hourly | nn | NN | 993 | 438 | 555 | 44.11% | 42.50% | 42.08% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 993 | 438 | 555 | 44.11% | 41.25% | 42.92% | 5.89 pp | -117 | 51 | -2.29 |
| Consolidated Hourly | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| BTC Daily | lstm | LSTM | 816 | 345 | 471 | 42.28% | 35.83% | 40.62% | 7.72 pp | -126 | 47 | -2.68 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 993 | 421 | 572 | 42.40% | 36.67% | 40.00% | 7.60 pp | -151 | 51 | -2.96 |
| BTC Daily | rf | RandomForest | 816 | 338 | 478 | 41.42% | 37.08% | 40.62% | 8.58 pp | -140 | 47 | -2.98 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 993 | 407 | 586 | 40.99% | 34.58% | 38.33% | 9.01 pp | -179 | 51 | -3.51 |
| BTC Daily | xgb | XGBoost | 826 | 324 | 502 | 39.23% | 36.67% | 35.83% | 10.77 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 993 | 471 | 522 | 47.43% | 49.17% | 46.25% | 2.57 pp | -51 | 51 | -1.00 |
| BTC Hourly | transformer | Transformer | 993 | 461 | 532 | 46.42% | 44.58% | 44.17% | 3.58 pp | -71 | 51 | -1.39 |
| BTC Hourly | nn | NN | 993 | 438 | 555 | 44.11% | 42.50% | 42.08% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | rf | RandomForest | 993 | 438 | 555 | 44.11% | 41.25% | 42.92% | 5.89 pp | -117 | 51 | -2.29 |
| BTC Hourly | lstm | LSTM | 993 | 421 | 572 | 42.40% | 36.67% | 40.00% | 7.60 pp | -151 | 51 | -2.96 |
| BTC Hourly | xgb | XGBoost | 993 | 407 | 586 | 40.99% | 34.58% | 38.33% | 9.01 pp | -179 | 51 | -3.51 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 816 | 392 | 424 | 48.04% | 45.42% | 46.88% | 1.96 pp | -32 | 47 | -0.68 |
| BTC Daily | nn | NN | 816 | 381 | 435 | 46.69% | 45.42% | 45.42% | 3.31 pp | -54 | 47 | -1.15 |
| BTC Daily | transformer | Transformer | 816 | 379 | 437 | 46.45% | 40.00% | 46.25% | 3.55 pp | -58 | 47 | -1.23 |
| BTC Daily | lstm | LSTM | 816 | 345 | 471 | 42.28% | 35.83% | 40.62% | 7.72 pp | -126 | 47 | -2.68 |
| BTC Daily | rf | RandomForest | 816 | 338 | 478 | 41.42% | 37.08% | 40.62% | 8.58 pp | -140 | 47 | -2.98 |
| BTC Daily | xgb | XGBoost | 826 | 324 | 502 | 39.23% | 36.67% | 35.83% | 10.77 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 588 | 285 | 303 | 48.47% | 47.92% | 47.50% | 1.53 pp | -18 | 55 | -0.33 |
| BTC Market Hours | nn | NN | 588 | 284 | 304 | 48.30% | 52.50% | 50.00% | 1.70 pp | -20 | 55 | -0.36 |
| BTC Market Hours | transformer | Transformer | 588 | 277 | 311 | 47.11% | 47.08% | 46.46% | 2.89 pp | -34 | 55 | -0.62 |
| BTC Market Hours | lstm | LSTM | 588 | 251 | 337 | 42.69% | 42.50% | 42.50% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | rf | RandomForest | 588 | 251 | 337 | 42.69% | 43.33% | 42.92% | 7.31 pp | -86 | 55 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 588 | 251 | 337 | 42.69% | 44.17% | 43.12% | 7.31 pp | -86 | 55 | -1.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 642 | 301 | 341 | 46.88% | 49.17% | 47.29% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | nn | NN | 642 | 301 | 341 | 46.88% | 48.33% | 48.12% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 642 | 301 | 341 | 46.88% | 49.17% | 47.71% | 3.12 pp | -40 | 55 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 642 | 266 | 376 | 41.43% | 42.08% | 40.83% | 8.57 pp | -110 | 55 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 642 | 264 | 378 | 41.12% | 42.92% | 40.42% | 8.88 pp | -114 | 55 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 642 | 263 | 379 | 40.97% | 42.08% | 40.83% | 9.03 pp | -116 | 55 | -2.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 229 | 110 | 119 | 48.03% | 48.03% | 48.03% | 1.97 pp | -9 | 15 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 229 | 106 | 123 | 46.29% | 46.29% | 46.29% | 3.71 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 229 | 97 | 132 | 42.36% | 42.36% | 42.36% | 7.64 pp | -35 | 15 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 15 | -2.47 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | nn | NN | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 7 | -3.14 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 81 | 35 | 46 | 43.21% | 43.21% | 43.21% | 6.79 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 81 | 34 | 47 | 41.98% | 41.98% | 41.98% | 8.02 pp | -13 | 7 | -1.86 |
| Consolidated Market Hours Daily | xgb | XGBoost | 81 | 33 | 48 | 40.74% | 40.74% | 40.74% | 9.26 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 81 | 31 | 50 | 38.27% | 38.27% | 38.27% | 11.73 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | nn | NN | 81 | 29 | 52 | 35.80% | 35.80% | 35.80% | 14.20 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
