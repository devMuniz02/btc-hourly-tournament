# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T04:31:54.236999+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1206 | 918 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1081 | 716 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 736 | 478 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 738 | 532 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 129 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 22:00:00+00:00 | 129 | 25 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 706 | 345 | 361 | 48.87% | 47.50% | 48.75% | 1.13 pp | -16 | 42 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 478 | 230 | 248 | 48.12% | 43.75% | 48.12% | 1.88 pp | -18 | 46 | -0.39 |
| Consolidated Hourly | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 478 | 225 | 253 | 47.07% | 48.33% | 47.07% | 2.93 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 478 | 223 | 255 | 46.65% | 41.67% | 46.65% | 3.35 pp | -32 | 46 | -0.70 |
| BTC Daily | transformer | Transformer | 706 | 338 | 368 | 47.88% | 47.08% | 50.00% | 2.12 pp | -30 | 42 | -0.71 |
| BTC Market Hours Daily | transformer | Transformer | 532 | 246 | 286 | 46.24% | 49.17% | 47.08% | 3.76 pp | -40 | 46 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 884 | 419 | 465 | 47.40% | 48.75% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 884 | 419 | 465 | 47.40% | 48.33% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 532 | 243 | 289 | 45.68% | 47.08% | 46.46% | 4.32 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | nn | NN | 532 | 243 | 289 | 45.68% | 42.92% | 46.25% | 4.32 pp | -46 | 46 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 706 | 327 | 379 | 46.32% | 43.33% | 48.12% | 3.68 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| BTC Market Hours | lstm | LSTM | 478 | 207 | 271 | 43.31% | 42.08% | 43.31% | 6.69 pp | -64 | 46 | -1.39 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours | rf | RandomForest | 478 | 204 | 274 | 42.68% | 42.08% | 42.68% | 7.32 pp | -70 | 46 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 478 | 194 | 284 | 40.59% | 39.58% | 40.59% | 9.41 pp | -90 | 46 | -1.96 |
| BTC Hourly | nn | NN | 884 | 396 | 488 | 44.80% | 45.42% | 43.33% | 5.20 pp | -92 | 47 | -1.96 |
| BTC Hourly | rf | RandomForest | 884 | 394 | 490 | 44.57% | 44.17% | 44.17% | 5.43 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 532 | 219 | 313 | 41.17% | 40.83% | 41.25% | 8.83 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 706 | 307 | 399 | 43.48% | 39.17% | 42.50% | 6.52 pp | -92 | 42 | -2.19 |
| Consolidated Hourly | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 532 | 212 | 320 | 39.85% | 38.75% | 39.17% | 10.15 pp | -108 | 46 | -2.35 |
| BTC Daily | rf | RandomForest | 706 | 303 | 403 | 42.92% | 41.67% | 43.33% | 7.08 pp | -100 | 42 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 532 | 211 | 321 | 39.66% | 37.08% | 40.42% | 10.34 pp | -110 | 46 | -2.39 |
| BTC Hourly | lstm | LSTM | 884 | 378 | 506 | 42.76% | 38.33% | 42.29% | 7.24 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 884 | 374 | 510 | 42.31% | 42.08% | 42.29% | 7.69 pp | -136 | 47 | -2.89 |
| BTC Daily | xgb | XGBoost | 716 | 281 | 435 | 39.25% | 33.75% | 39.17% | 10.75 pp | -154 | 42 | -3.67 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 884 | 419 | 465 | 47.40% | 48.75% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 884 | 419 | 465 | 47.40% | 48.33% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 884 | 396 | 488 | 44.80% | 45.42% | 43.33% | 5.20 pp | -92 | 47 | -1.96 |
| BTC Hourly | rf | RandomForest | 884 | 394 | 490 | 44.57% | 44.17% | 44.17% | 5.43 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 884 | 378 | 506 | 42.76% | 38.33% | 42.29% | 7.24 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 884 | 374 | 510 | 42.31% | 42.08% | 42.29% | 7.69 pp | -136 | 47 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 706 | 345 | 361 | 48.87% | 47.50% | 48.75% | 1.13 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 706 | 338 | 368 | 47.88% | 47.08% | 50.00% | 2.12 pp | -30 | 42 | -0.71 |
| BTC Daily | nn | NN | 706 | 327 | 379 | 46.32% | 43.33% | 48.12% | 3.68 pp | -52 | 42 | -1.24 |
| BTC Daily | lstm | LSTM | 706 | 307 | 399 | 43.48% | 39.17% | 42.50% | 6.52 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 706 | 303 | 403 | 42.92% | 41.67% | 43.33% | 7.08 pp | -100 | 42 | -2.38 |
| BTC Daily | xgb | XGBoost | 716 | 281 | 435 | 39.25% | 33.75% | 39.17% | 10.75 pp | -154 | 42 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 478 | 230 | 248 | 48.12% | 43.75% | 48.12% | 1.88 pp | -18 | 46 | -0.39 |
| BTC Market Hours | nn | NN | 478 | 225 | 253 | 47.07% | 48.33% | 47.07% | 2.93 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 478 | 223 | 255 | 46.65% | 41.67% | 46.65% | 3.35 pp | -32 | 46 | -0.70 |
| BTC Market Hours | lstm | LSTM | 478 | 207 | 271 | 43.31% | 42.08% | 43.31% | 6.69 pp | -64 | 46 | -1.39 |
| BTC Market Hours | rf | RandomForest | 478 | 204 | 274 | 42.68% | 42.08% | 42.68% | 7.32 pp | -70 | 46 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 478 | 194 | 284 | 40.59% | 39.58% | 40.59% | 9.41 pp | -90 | 46 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 532 | 246 | 286 | 46.24% | 49.17% | 47.08% | 3.76 pp | -40 | 46 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 532 | 243 | 289 | 45.68% | 47.08% | 46.46% | 4.32 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | nn | NN | 532 | 243 | 289 | 45.68% | 42.92% | 46.25% | 4.32 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 532 | 219 | 313 | 41.17% | 40.83% | 41.25% | 8.83 pp | -94 | 46 | -2.04 |
| BTC Market Hours Daily | xgb | XGBoost | 532 | 212 | 320 | 39.85% | 38.75% | 39.17% | 10.15 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | lstm | LSTM | 532 | 211 | 321 | 39.66% | 37.08% | 40.42% | 10.34 pp | -110 | 46 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 68 | 61 | 52.71% | 52.71% | 52.71% | 2.71 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 59 | 70 | 45.74% | 45.74% | 45.74% | 4.26 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 58 | 71 | 44.96% | 44.96% | 44.96% | 5.04 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 53 | 76 | 41.09% | 41.09% | 41.09% | 8.91 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
