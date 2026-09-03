# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T01:39:02.081833+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1204 | 916 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1079 | 714 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 734 | 476 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 736 | 530 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 127 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 127 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 24 | 103 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 21:00:00+00:00 | 127 | 24 | 103 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 476 | 230 | 246 | 48.32% | 43.75% | 48.32% | 1.68 pp | -16 | 46 | -0.35 |
| BTC Daily | mlp_sklearn | MLPClassifier | 704 | 344 | 360 | 48.86% | 47.08% | 48.75% | 1.14 pp | -16 | 42 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| BTC Market Hours | nn | NN | 476 | 224 | 252 | 47.06% | 47.92% | 47.06% | 2.94 pp | -28 | 46 | -0.61 |
| BTC Daily | transformer | Transformer | 704 | 338 | 366 | 48.01% | 47.08% | 50.00% | 1.99 pp | -28 | 42 | -0.67 |
| BTC Market Hours | transformer | Transformer | 476 | 221 | 255 | 46.43% | 40.83% | 46.43% | 3.57 pp | -34 | 46 | -0.74 |
| BTC Market Hours Daily | transformer | Transformer | 530 | 244 | 286 | 46.04% | 48.33% | 46.88% | 3.96 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 530 | 243 | 287 | 45.85% | 47.50% | 46.67% | 4.15 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | nn | NN | 530 | 243 | 287 | 45.85% | 43.33% | 46.67% | 4.15 pp | -44 | 46 | -0.96 |
| BTC Hourly | transformer | Transformer | 882 | 418 | 464 | 47.39% | 48.75% | 47.71% | 2.61 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 882 | 417 | 465 | 47.28% | 47.92% | 47.50% | 2.72 pp | -48 | 47 | -1.02 |
| Consolidated Hourly | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 704 | 326 | 378 | 46.31% | 42.92% | 48.33% | 3.69 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| BTC Market Hours | lstm | LSTM | 476 | 205 | 271 | 43.07% | 41.25% | 43.07% | 6.93 pp | -66 | 46 | -1.43 |
| BTC Market Hours | rf | RandomForest | 476 | 204 | 272 | 42.86% | 42.08% | 42.86% | 7.14 pp | -68 | 46 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 476 | 194 | 282 | 40.76% | 39.58% | 40.76% | 9.24 pp | -88 | 46 | -1.91 |
| BTC Hourly | nn | NN | 882 | 396 | 486 | 44.90% | 45.83% | 43.54% | 5.10 pp | -90 | 47 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 530 | 219 | 311 | 41.32% | 41.25% | 41.46% | 8.68 pp | -92 | 46 | -2.00 |
| BTC Hourly | rf | RandomForest | 882 | 392 | 490 | 44.44% | 44.17% | 43.96% | 5.56 pp | -98 | 47 | -2.09 |
| BTC Daily | lstm | LSTM | 704 | 306 | 398 | 43.47% | 38.75% | 42.50% | 6.53 pp | -92 | 42 | -2.19 |
| Consolidated Hourly | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 530 | 211 | 319 | 39.81% | 37.08% | 40.62% | 10.19 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 530 | 211 | 319 | 39.81% | 38.33% | 39.17% | 10.19 pp | -108 | 46 | -2.35 |
| BTC Daily | rf | RandomForest | 704 | 302 | 402 | 42.90% | 41.25% | 43.33% | 7.10 pp | -100 | 42 | -2.38 |
| BTC Hourly | lstm | LSTM | 882 | 376 | 506 | 42.63% | 37.92% | 41.88% | 7.37 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 882 | 373 | 509 | 42.29% | 41.67% | 42.50% | 7.71 pp | -136 | 47 | -2.89 |
| BTC Daily | xgb | XGBoost | 714 | 281 | 433 | 39.36% | 34.17% | 39.17% | 10.64 pp | -152 | 42 | -3.62 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 882 | 418 | 464 | 47.39% | 48.75% | 47.71% | 2.61 pp | -46 | 47 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 882 | 417 | 465 | 47.28% | 47.92% | 47.50% | 2.72 pp | -48 | 47 | -1.02 |
| BTC Hourly | nn | NN | 882 | 396 | 486 | 44.90% | 45.83% | 43.54% | 5.10 pp | -90 | 47 | -1.91 |
| BTC Hourly | rf | RandomForest | 882 | 392 | 490 | 44.44% | 44.17% | 43.96% | 5.56 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 882 | 376 | 506 | 42.63% | 37.92% | 41.88% | 7.37 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 882 | 373 | 509 | 42.29% | 41.67% | 42.50% | 7.71 pp | -136 | 47 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 704 | 344 | 360 | 48.86% | 47.08% | 48.75% | 1.14 pp | -16 | 42 | -0.38 |
| BTC Daily | transformer | Transformer | 704 | 338 | 366 | 48.01% | 47.08% | 50.00% | 1.99 pp | -28 | 42 | -0.67 |
| BTC Daily | nn | NN | 704 | 326 | 378 | 46.31% | 42.92% | 48.33% | 3.69 pp | -52 | 42 | -1.24 |
| BTC Daily | lstm | LSTM | 704 | 306 | 398 | 43.47% | 38.75% | 42.50% | 6.53 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 704 | 302 | 402 | 42.90% | 41.25% | 43.33% | 7.10 pp | -100 | 42 | -2.38 |
| BTC Daily | xgb | XGBoost | 714 | 281 | 433 | 39.36% | 34.17% | 39.17% | 10.64 pp | -152 | 42 | -3.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 476 | 230 | 246 | 48.32% | 43.75% | 48.32% | 1.68 pp | -16 | 46 | -0.35 |
| BTC Market Hours | nn | NN | 476 | 224 | 252 | 47.06% | 47.92% | 47.06% | 2.94 pp | -28 | 46 | -0.61 |
| BTC Market Hours | transformer | Transformer | 476 | 221 | 255 | 46.43% | 40.83% | 46.43% | 3.57 pp | -34 | 46 | -0.74 |
| BTC Market Hours | lstm | LSTM | 476 | 205 | 271 | 43.07% | 41.25% | 43.07% | 6.93 pp | -66 | 46 | -1.43 |
| BTC Market Hours | rf | RandomForest | 476 | 204 | 272 | 42.86% | 42.08% | 42.86% | 7.14 pp | -68 | 46 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 476 | 194 | 282 | 40.76% | 39.58% | 40.76% | 9.24 pp | -88 | 46 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 530 | 244 | 286 | 46.04% | 48.33% | 46.88% | 3.96 pp | -42 | 46 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 530 | 243 | 287 | 45.85% | 47.50% | 46.67% | 4.15 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | nn | NN | 530 | 243 | 287 | 45.85% | 43.33% | 46.67% | 4.15 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 530 | 219 | 311 | 41.32% | 41.25% | 41.46% | 8.68 pp | -92 | 46 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 530 | 211 | 319 | 39.81% | 37.08% | 40.62% | 10.19 pp | -108 | 46 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 530 | 211 | 319 | 39.81% | 38.33% | 39.17% | 10.19 pp | -108 | 46 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 127 | 67 | 60 | 52.76% | 52.76% | 52.76% | 2.76 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 127 | 63 | 64 | 49.61% | 49.61% | 49.61% | 0.39 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 127 | 52 | 75 | 40.94% | 40.94% | 40.94% | 9.06 pp | -23 | 10 | -2.30 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 24 | 8 | 16 | 33.33% | 33.33% | 33.33% | 16.67 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 2 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
