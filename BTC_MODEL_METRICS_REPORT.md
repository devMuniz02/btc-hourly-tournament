# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T12:12:18.436002+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1260 | 972 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1135 | 770 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 829 | 532 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 831 | 586 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 178 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 23:00:00+00:00 | 178 | 52 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 532 | 259 | 273 | 48.68% | 45.42% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 532 | 255 | 277 | 47.93% | 48.75% | 48.75% | 2.07 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 760 | 369 | 391 | 48.55% | 47.92% | 48.54% | 1.45 pp | -22 | 45 | -0.49 |
| BTC Market Hours | nn | NN | 532 | 253 | 279 | 47.56% | 51.25% | 49.17% | 2.44 pp | -26 | 50 | -0.52 |
| BTC Market Hours Daily | transformer | Transformer | 586 | 278 | 308 | 47.44% | 50.83% | 48.54% | 2.56 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 586 | 273 | 313 | 46.59% | 46.67% | 47.92% | 3.41 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 586 | 272 | 314 | 46.42% | 51.25% | 47.29% | 3.58 pp | -42 | 50 | -0.84 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 938 | 447 | 491 | 47.65% | 49.58% | 46.88% | 2.35 pp | -44 | 49 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Daily | transformer | Transformer | 760 | 357 | 403 | 46.97% | 42.92% | 47.92% | 3.03 pp | -46 | 45 | -1.02 |
| BTC Hourly | transformer | Transformer | 938 | 440 | 498 | 46.91% | 46.25% | 45.21% | 3.09 pp | -58 | 49 | -1.18 |
| BTC Daily | nn | NN | 760 | 352 | 408 | 46.32% | 45.00% | 46.25% | 3.68 pp | -56 | 45 | -1.24 |
| BTC Market Hours | lstm | LSTM | 532 | 230 | 302 | 43.23% | 42.50% | 44.17% | 6.77 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 532 | 229 | 303 | 43.05% | 43.75% | 43.75% | 6.95 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 532 | 221 | 311 | 41.54% | 42.92% | 42.08% | 8.46 pp | -90 | 50 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 586 | 243 | 343 | 41.47% | 44.17% | 41.25% | 8.53 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 938 | 417 | 521 | 44.46% | 44.17% | 43.75% | 5.54 pp | -104 | 49 | -2.12 |
| BTC Hourly | nn | NN | 938 | 416 | 522 | 44.35% | 42.08% | 42.29% | 5.65 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 586 | 238 | 348 | 40.61% | 40.00% | 40.42% | 9.39 pp | -110 | 50 | -2.20 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 586 | 233 | 353 | 39.76% | 41.25% | 38.96% | 10.24 pp | -120 | 50 | -2.40 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 760 | 322 | 438 | 42.37% | 35.83% | 40.62% | 7.63 pp | -116 | 45 | -2.58 |
| BTC Daily | rf | RandomForest | 760 | 317 | 443 | 41.71% | 37.92% | 41.88% | 8.29 pp | -126 | 45 | -2.80 |
| BTC Hourly | lstm | LSTM | 938 | 400 | 538 | 42.64% | 36.67% | 41.67% | 7.36 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 938 | 394 | 544 | 42.00% | 41.25% | 40.42% | 8.00 pp | -150 | 49 | -3.06 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 770 | 302 | 468 | 39.22% | 35.42% | 36.88% | 10.78 pp | -166 | 45 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 938 | 447 | 491 | 47.65% | 49.58% | 46.88% | 2.35 pp | -44 | 49 | -0.90 |
| BTC Hourly | transformer | Transformer | 938 | 440 | 498 | 46.91% | 46.25% | 45.21% | 3.09 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 938 | 417 | 521 | 44.46% | 44.17% | 43.75% | 5.54 pp | -104 | 49 | -2.12 |
| BTC Hourly | nn | NN | 938 | 416 | 522 | 44.35% | 42.08% | 42.29% | 5.65 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 938 | 400 | 538 | 42.64% | 36.67% | 41.67% | 7.36 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 938 | 394 | 544 | 42.00% | 41.25% | 40.42% | 8.00 pp | -150 | 49 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 760 | 369 | 391 | 48.55% | 47.92% | 48.54% | 1.45 pp | -22 | 45 | -0.49 |
| BTC Daily | transformer | Transformer | 760 | 357 | 403 | 46.97% | 42.92% | 47.92% | 3.03 pp | -46 | 45 | -1.02 |
| BTC Daily | nn | NN | 760 | 352 | 408 | 46.32% | 45.00% | 46.25% | 3.68 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 760 | 322 | 438 | 42.37% | 35.83% | 40.62% | 7.63 pp | -116 | 45 | -2.58 |
| BTC Daily | rf | RandomForest | 760 | 317 | 443 | 41.71% | 37.92% | 41.88% | 8.29 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 770 | 302 | 468 | 39.22% | 35.42% | 36.88% | 10.78 pp | -166 | 45 | -3.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 532 | 259 | 273 | 48.68% | 45.42% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 532 | 255 | 277 | 47.93% | 48.75% | 48.75% | 2.07 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 532 | 253 | 279 | 47.56% | 51.25% | 49.17% | 2.44 pp | -26 | 50 | -0.52 |
| BTC Market Hours | lstm | LSTM | 532 | 230 | 302 | 43.23% | 42.50% | 44.17% | 6.77 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 532 | 229 | 303 | 43.05% | 43.75% | 43.75% | 6.95 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 532 | 221 | 311 | 41.54% | 42.92% | 42.08% | 8.46 pp | -90 | 50 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 586 | 278 | 308 | 47.44% | 50.83% | 48.54% | 2.56 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 586 | 273 | 313 | 46.59% | 46.67% | 47.92% | 3.41 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 586 | 272 | 314 | 46.42% | 51.25% | 47.29% | 3.58 pp | -42 | 50 | -0.84 |
| BTC Market Hours Daily | rf | RandomForest | 586 | 243 | 343 | 41.47% | 44.17% | 41.25% | 8.53 pp | -100 | 50 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 586 | 238 | 348 | 40.61% | 40.00% | 40.42% | 9.39 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 586 | 233 | 353 | 39.76% | 41.25% | 38.96% | 10.24 pp | -120 | 50 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 88 | 90 | 49.44% | 49.44% | 49.44% | 0.56 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 83 | 95 | 46.63% | 46.63% | 46.63% | 3.37 pp | -12 | 12 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 12 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 12 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
