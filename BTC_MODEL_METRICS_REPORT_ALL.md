# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T03:22:51.409204+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1221 | 933 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1097 | 732 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 765 | 494 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 767 | 548 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 494 | 238 | 256 | 48.18% | 44.17% | 47.92% | 1.82 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 494 | 233 | 261 | 47.17% | 50.00% | 47.71% | 2.83 pp | -28 | 47 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 722 | 348 | 374 | 48.20% | 46.67% | 47.92% | 1.80 pp | -26 | 43 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 494 | 231 | 263 | 46.76% | 43.33% | 47.29% | 3.24 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 722 | 346 | 376 | 47.92% | 47.08% | 50.42% | 2.08 pp | -30 | 43 | -0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 548 | 254 | 294 | 46.35% | 48.75% | 47.08% | 3.65 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 548 | 253 | 295 | 46.17% | 45.00% | 47.29% | 3.83 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 548 | 251 | 297 | 45.80% | 48.33% | 46.88% | 4.20 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Daily | nn | NN | 722 | 335 | 387 | 46.40% | 44.58% | 47.71% | 3.60 pp | -52 | 43 | -1.21 |
| BTC Market Hours | rf | RandomForest | 494 | 214 | 280 | 43.32% | 44.17% | 43.54% | 6.68 pp | -66 | 47 | -1.40 |
| BTC Market Hours | lstm | LSTM | 494 | 213 | 281 | 43.12% | 40.83% | 43.12% | 6.88 pp | -68 | 47 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 494 | 203 | 291 | 41.09% | 40.83% | 41.04% | 8.91 pp | -88 | 47 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 548 | 228 | 320 | 41.61% | 42.08% | 41.25% | 8.39 pp | -92 | 47 | -1.96 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 548 | 221 | 327 | 40.33% | 38.33% | 40.83% | 9.67 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 722 | 312 | 410 | 43.21% | 37.92% | 42.08% | 6.79 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 548 | 220 | 328 | 40.15% | 41.25% | 39.38% | 9.85 pp | -108 | 47 | -2.30 |
| BTC Daily | rf | RandomForest | 722 | 309 | 413 | 42.80% | 41.25% | 43.54% | 7.20 pp | -104 | 43 | -2.42 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 732 | 291 | 441 | 39.75% | 36.67% | 38.75% | 10.25 pp | -150 | 43 | -3.49 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 722 | 348 | 374 | 48.20% | 46.67% | 47.92% | 1.80 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 722 | 346 | 376 | 47.92% | 47.08% | 50.42% | 2.08 pp | -30 | 43 | -0.70 |
| BTC Daily | nn | NN | 722 | 335 | 387 | 46.40% | 44.58% | 47.71% | 3.60 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 722 | 312 | 410 | 43.21% | 37.92% | 42.08% | 6.79 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 722 | 309 | 413 | 42.80% | 41.25% | 43.54% | 7.20 pp | -104 | 43 | -2.42 |
| BTC Daily | xgb | XGBoost | 732 | 291 | 441 | 39.75% | 36.67% | 38.75% | 10.25 pp | -150 | 43 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 494 | 238 | 256 | 48.18% | 44.17% | 47.92% | 1.82 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 494 | 233 | 261 | 47.17% | 50.00% | 47.71% | 2.83 pp | -28 | 47 | -0.60 |
| BTC Market Hours | transformer | Transformer | 494 | 231 | 263 | 46.76% | 43.33% | 47.29% | 3.24 pp | -32 | 47 | -0.68 |
| BTC Market Hours | rf | RandomForest | 494 | 214 | 280 | 43.32% | 44.17% | 43.54% | 6.68 pp | -66 | 47 | -1.40 |
| BTC Market Hours | lstm | LSTM | 494 | 213 | 281 | 43.12% | 40.83% | 43.12% | 6.88 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 494 | 203 | 291 | 41.09% | 40.83% | 41.04% | 8.91 pp | -88 | 47 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 548 | 254 | 294 | 46.35% | 48.75% | 47.08% | 3.65 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 548 | 253 | 295 | 46.17% | 45.00% | 47.29% | 3.83 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 548 | 251 | 297 | 45.80% | 48.33% | 46.88% | 4.20 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 548 | 228 | 320 | 41.61% | 42.08% | 41.25% | 8.39 pp | -92 | 47 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 548 | 221 | 327 | 40.33% | 38.33% | 40.83% | 9.67 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 548 | 220 | 328 | 40.15% | 41.25% | 39.38% | 9.85 pp | -108 | 47 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
