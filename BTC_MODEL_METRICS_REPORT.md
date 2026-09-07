# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T10:41:29.153245+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 193 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 548 | 277 | 271 | 50.55% | 47.08% | 50.00% | 0.55 pp | 6 | 52 | 0.12 |
| BTC Market Hours | nn | NN | 548 | 271 | 277 | 49.45% | 51.67% | 51.04% | 0.55 pp | -6 | 52 | -0.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 602 | 287 | 315 | 47.67% | 47.50% | 47.92% | 2.33 pp | -28 | 51 | -0.55 |
| BTC Daily | mlp_sklearn | MLPClassifier | 776 | 373 | 403 | 48.07% | 45.83% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 602 | 283 | 319 | 47.01% | 47.92% | 46.67% | 2.99 pp | -36 | 51 | -0.71 |
| BTC Market Hours | transformer | Transformer | 548 | 254 | 294 | 46.35% | 45.42% | 47.08% | 3.65 pp | -40 | 52 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 953 | 454 | 499 | 47.64% | 49.58% | 47.08% | 2.36 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 602 | 278 | 324 | 46.18% | 47.08% | 47.50% | 3.82 pp | -46 | 51 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| BTC Market Hours | rf | RandomForest | 548 | 244 | 304 | 44.53% | 47.08% | 44.38% | 5.47 pp | -60 | 52 | -1.15 |
| BTC Daily | transformer | Transformer | 776 | 362 | 414 | 46.65% | 42.08% | 46.88% | 3.35 pp | -52 | 45 | -1.16 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 776 | 359 | 417 | 46.26% | 44.17% | 45.21% | 3.74 pp | -58 | 45 | -1.29 |
| BTC Market Hours Daily | rf | RandomForest | 602 | 267 | 335 | 44.35% | 46.25% | 43.75% | 5.65 pp | -68 | 51 | -1.33 |
| BTC Hourly | transformer | Transformer | 953 | 443 | 510 | 46.48% | 44.17% | 44.17% | 3.52 pp | -67 | 50 | -1.34 |
| Consolidated Hourly | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 548 | 227 | 321 | 41.42% | 42.50% | 41.46% | 8.58 pp | -94 | 52 | -1.81 |
| Consolidated Hourly | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |
| BTC Market Hours | lstm | LSTM | 548 | 226 | 322 | 41.24% | 35.83% | 41.04% | 8.76 pp | -96 | 52 | -1.85 |
| BTC Market Hours Daily | xgb | XGBoost | 602 | 252 | 350 | 41.86% | 42.08% | 41.46% | 8.14 pp | -98 | 51 | -1.92 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| BTC Hourly | rf | RandomForest | 953 | 423 | 530 | 44.39% | 43.33% | 43.33% | 5.61 pp | -107 | 50 | -2.14 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Hourly | nn | NN | 953 | 421 | 532 | 44.18% | 42.08% | 42.50% | 5.82 pp | -111 | 50 | -2.22 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 602 | 237 | 365 | 39.37% | 35.42% | 38.75% | 10.63 pp | -128 | 51 | -2.51 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 776 | 327 | 449 | 42.14% | 35.42% | 39.79% | 7.86 pp | -122 | 45 | -2.71 |
| BTC Hourly | lstm | LSTM | 953 | 407 | 546 | 42.71% | 37.08% | 42.08% | 7.29 pp | -139 | 50 | -2.78 |
| BTC Daily | rf | RandomForest | 776 | 325 | 451 | 41.88% | 38.75% | 41.88% | 8.12 pp | -126 | 45 | -2.80 |
| BTC Hourly | xgb | XGBoost | 953 | 397 | 556 | 41.66% | 39.17% | 40.21% | 8.34 pp | -159 | 50 | -3.18 |
| BTC Daily | xgb | XGBoost | 786 | 307 | 479 | 39.06% | 35.42% | 36.25% | 10.94 pp | -172 | 45 | -3.82 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 776 | 373 | 403 | 48.07% | 45.83% | 47.29% | 1.93 pp | -30 | 45 | -0.67 |
| BTC Daily | transformer | Transformer | 776 | 362 | 414 | 46.65% | 42.08% | 46.88% | 3.35 pp | -52 | 45 | -1.16 |
| BTC Daily | nn | NN | 776 | 359 | 417 | 46.26% | 44.17% | 45.21% | 3.74 pp | -58 | 45 | -1.29 |
| BTC Daily | lstm | LSTM | 776 | 327 | 449 | 42.14% | 35.42% | 39.79% | 7.86 pp | -122 | 45 | -2.71 |
| BTC Daily | rf | RandomForest | 776 | 325 | 451 | 41.88% | 38.75% | 41.88% | 8.12 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 786 | 307 | 479 | 39.06% | 35.42% | 36.25% | 10.94 pp | -172 | 45 | -3.82 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 548 | 277 | 271 | 50.55% | 47.08% | 50.00% | 0.55 pp | 6 | 52 | 0.12 |
| BTC Market Hours | nn | NN | 548 | 271 | 277 | 49.45% | 51.67% | 51.04% | 0.55 pp | -6 | 52 | -0.12 |
| BTC Market Hours | transformer | Transformer | 548 | 254 | 294 | 46.35% | 45.42% | 47.08% | 3.65 pp | -40 | 52 | -0.77 |
| BTC Market Hours | rf | RandomForest | 548 | 244 | 304 | 44.53% | 47.08% | 44.38% | 5.47 pp | -60 | 52 | -1.15 |
| BTC Market Hours | xgb | XGBoost | 548 | 227 | 321 | 41.42% | 42.50% | 41.46% | 8.58 pp | -94 | 52 | -1.81 |
| BTC Market Hours | lstm | LSTM | 548 | 226 | 322 | 41.24% | 35.83% | 41.04% | 8.76 pp | -96 | 52 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 602 | 287 | 315 | 47.67% | 47.50% | 47.92% | 2.33 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 602 | 283 | 319 | 47.01% | 47.92% | 46.67% | 2.99 pp | -36 | 51 | -0.71 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 602 | 278 | 324 | 46.18% | 47.08% | 47.50% | 3.82 pp | -46 | 51 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 602 | 267 | 335 | 44.35% | 46.25% | 43.75% | 5.65 pp | -68 | 51 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 602 | 252 | 350 | 41.86% | 42.08% | 41.46% | 8.14 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 602 | 237 | 365 | 39.37% | 35.42% | 38.75% | 10.63 pp | -128 | 51 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Hourly | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 192 | 95 | 97 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 13 | -0.15 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 192 | 94 | 98 | 48.96% | 48.96% | 48.96% | 1.04 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 13 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 192 | 87 | 105 | 45.31% | 45.31% | 45.31% | 4.69 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 192 | 84 | 108 | 43.75% | 43.75% | 43.75% | 6.25 pp | -24 | 13 | -1.85 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
