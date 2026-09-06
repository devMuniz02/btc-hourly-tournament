# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T07:51:55.113487+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1257 | 969 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1133 | 768 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 827 | 530 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 828 | 583 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 175 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 21:00:00+00:00 | 175 | 50 | 125 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 530 | 258 | 272 | 48.68% | 45.83% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 530 | 254 | 276 | 47.92% | 48.33% | 48.54% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 758 | 369 | 389 | 48.68% | 47.92% | 48.96% | 1.32 pp | -20 | 44 | -0.45 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours | nn | NN | 530 | 251 | 279 | 47.36% | 50.83% | 48.96% | 2.64 pp | -28 | 50 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 583 | 277 | 306 | 47.51% | 51.25% | 48.75% | 2.49 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 583 | 272 | 311 | 46.66% | 46.25% | 47.92% | 3.34 pp | -39 | 50 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 935 | 447 | 488 | 47.81% | 50.00% | 47.50% | 2.19 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 583 | 270 | 313 | 46.31% | 51.25% | 47.08% | 3.69 pp | -43 | 50 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| BTC Daily | transformer | Transformer | 758 | 357 | 401 | 47.10% | 43.33% | 48.33% | 2.90 pp | -44 | 44 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 935 | 438 | 497 | 46.84% | 45.83% | 45.42% | 3.16 pp | -59 | 49 | -1.20 |
| BTC Daily | nn | NN | 758 | 351 | 407 | 46.31% | 44.58% | 46.04% | 3.69 pp | -56 | 44 | -1.27 |
| BTC Market Hours | lstm | LSTM | 530 | 229 | 301 | 43.21% | 42.08% | 44.17% | 6.79 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 530 | 228 | 302 | 43.02% | 44.17% | 43.75% | 6.98 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 530 | 219 | 311 | 41.32% | 42.92% | 41.88% | 8.68 pp | -92 | 50 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 583 | 242 | 341 | 41.51% | 43.75% | 41.25% | 8.49 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 935 | 417 | 518 | 44.60% | 44.58% | 44.38% | 5.40 pp | -101 | 49 | -2.06 |
| BTC Hourly | nn | NN | 935 | 415 | 520 | 44.39% | 42.50% | 42.50% | 5.61 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 583 | 236 | 347 | 40.48% | 40.00% | 40.42% | 9.52 pp | -111 | 50 | -2.22 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 583 | 232 | 351 | 39.79% | 40.83% | 38.96% | 10.21 pp | -119 | 50 | -2.38 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 758 | 321 | 437 | 42.35% | 35.83% | 40.42% | 7.65 pp | -116 | 44 | -2.64 |
| BTC Hourly | lstm | LSTM | 935 | 400 | 535 | 42.78% | 37.50% | 42.08% | 7.22 pp | -135 | 49 | -2.76 |
| BTC Daily | rf | RandomForest | 758 | 316 | 442 | 41.69% | 37.92% | 41.67% | 8.31 pp | -126 | 44 | -2.86 |
| BTC Hourly | xgb | XGBoost | 935 | 394 | 541 | 42.14% | 41.25% | 41.04% | 7.86 pp | -147 | 49 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 768 | 301 | 467 | 39.19% | 35.42% | 36.88% | 10.81 pp | -166 | 44 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 935 | 447 | 488 | 47.81% | 50.00% | 47.50% | 2.19 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 935 | 438 | 497 | 46.84% | 45.83% | 45.42% | 3.16 pp | -59 | 49 | -1.20 |
| BTC Hourly | rf | RandomForest | 935 | 417 | 518 | 44.60% | 44.58% | 44.38% | 5.40 pp | -101 | 49 | -2.06 |
| BTC Hourly | nn | NN | 935 | 415 | 520 | 44.39% | 42.50% | 42.50% | 5.61 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 935 | 400 | 535 | 42.78% | 37.50% | 42.08% | 7.22 pp | -135 | 49 | -2.76 |
| BTC Hourly | xgb | XGBoost | 935 | 394 | 541 | 42.14% | 41.25% | 41.04% | 7.86 pp | -147 | 49 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 758 | 369 | 389 | 48.68% | 47.92% | 48.96% | 1.32 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 758 | 357 | 401 | 47.10% | 43.33% | 48.33% | 2.90 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 758 | 351 | 407 | 46.31% | 44.58% | 46.04% | 3.69 pp | -56 | 44 | -1.27 |
| BTC Daily | lstm | LSTM | 758 | 321 | 437 | 42.35% | 35.83% | 40.42% | 7.65 pp | -116 | 44 | -2.64 |
| BTC Daily | rf | RandomForest | 758 | 316 | 442 | 41.69% | 37.92% | 41.67% | 8.31 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 768 | 301 | 467 | 39.19% | 35.42% | 36.88% | 10.81 pp | -166 | 44 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 530 | 258 | 272 | 48.68% | 45.83% | 48.75% | 1.32 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 530 | 254 | 276 | 47.92% | 48.33% | 48.54% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 530 | 251 | 279 | 47.36% | 50.83% | 48.96% | 2.64 pp | -28 | 50 | -0.56 |
| BTC Market Hours | lstm | LSTM | 530 | 229 | 301 | 43.21% | 42.08% | 44.17% | 6.79 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 530 | 228 | 302 | 43.02% | 44.17% | 43.75% | 6.98 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 530 | 219 | 311 | 41.32% | 42.92% | 41.88% | 8.68 pp | -92 | 50 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 583 | 277 | 306 | 47.51% | 51.25% | 48.75% | 2.49 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 583 | 272 | 311 | 46.66% | 46.25% | 47.92% | 3.34 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 583 | 270 | 313 | 46.31% | 51.25% | 47.08% | 3.69 pp | -43 | 50 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 583 | 242 | 341 | 41.51% | 43.75% | 41.25% | 8.49 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 583 | 236 | 347 | 40.48% | 40.00% | 40.42% | 9.52 pp | -111 | 50 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 583 | 232 | 351 | 39.79% | 40.83% | 38.96% | 10.21 pp | -119 | 50 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 87 | 88 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 82 | 93 | 46.86% | 46.86% | 46.86% | 3.14 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 76 | 99 | 43.43% | 43.43% | 43.43% | 6.57 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
