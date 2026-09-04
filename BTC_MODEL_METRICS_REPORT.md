# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T00:29:33.200447+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1219 | 931 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1095 | 730 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 23:00:00+00:00 | 762 | 492 | 269 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 23:00:00+00:00 | 764 | 546 | 216 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 140 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 140 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 140 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T15:00:00+00:00 | 141 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 11 | -0.18 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 492 | 237 | 255 | 48.17% | 44.58% | 48.12% | 1.83 pp | -18 | 47 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 720 | 348 | 372 | 48.33% | 46.67% | 48.12% | 1.67 pp | -24 | 43 | -0.56 |
| BTC Market Hours | nn | NN | 492 | 232 | 260 | 47.15% | 50.00% | 47.71% | 2.85 pp | -28 | 47 | -0.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 492 | 230 | 262 | 46.75% | 43.33% | 47.29% | 3.25 pp | -32 | 47 | -0.68 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 140 | 66 | 74 | 47.14% | 47.14% | 47.14% | 2.86 pp | -8 | 11 | -0.73 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 140 | 66 | 74 | 47.14% | 47.14% | 47.14% | 2.86 pp | -8 | 11 | -0.73 |
| BTC Daily | transformer | Transformer | 720 | 344 | 376 | 47.78% | 46.25% | 50.21% | 2.22 pp | -32 | 43 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 897 | 429 | 468 | 47.83% | 50.83% | 48.75% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 546 | 253 | 293 | 46.34% | 48.75% | 47.29% | 3.66 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 546 | 251 | 295 | 45.97% | 44.17% | 47.08% | 4.03 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 546 | 250 | 296 | 45.79% | 47.92% | 46.88% | 4.21 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 897 | 424 | 473 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| Consolidated Hourly | lstm | LSTM | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| BTC Daily | nn | NN | 720 | 334 | 386 | 46.39% | 44.17% | 47.92% | 3.61 pp | -52 | 43 | -1.21 |
| Consolidated Hourly | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| BTC Market Hours | lstm | LSTM | 492 | 212 | 280 | 43.09% | 41.25% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 492 | 212 | 280 | 43.09% | 43.33% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 492 | 201 | 291 | 40.85% | 40.00% | 40.83% | 9.15 pp | -90 | 47 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 546 | 227 | 319 | 41.58% | 41.67% | 41.25% | 8.42 pp | -92 | 47 | -1.96 |
| Consolidated Hourly | transformer | Transformer | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 11 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 11 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 897 | 399 | 498 | 44.48% | 44.17% | 42.50% | 5.52 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 897 | 399 | 498 | 44.48% | 45.00% | 44.17% | 5.52 pp | -99 | 47 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 546 | 220 | 326 | 40.29% | 37.92% | 40.83% | 9.71 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 720 | 311 | 409 | 43.19% | 37.50% | 41.88% | 6.81 pp | -98 | 43 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 546 | 219 | 327 | 40.11% | 40.83% | 39.58% | 9.89 pp | -108 | 47 | -2.30 |
| BTC Daily | rf | RandomForest | 720 | 307 | 413 | 42.64% | 40.42% | 43.54% | 7.36 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 897 | 383 | 514 | 42.70% | 39.17% | 42.29% | 7.30 pp | -131 | 47 | -2.79 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 897 | 377 | 520 | 42.03% | 42.50% | 41.88% | 7.97 pp | -143 | 47 | -3.04 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 730 | 290 | 440 | 39.73% | 36.25% | 38.75% | 10.27 pp | -150 | 43 | -3.49 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 897 | 429 | 468 | 47.83% | 50.83% | 48.75% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Hourly | transformer | Transformer | 897 | 424 | 473 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 897 | 399 | 498 | 44.48% | 44.17% | 42.50% | 5.52 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 897 | 399 | 498 | 44.48% | 45.00% | 44.17% | 5.52 pp | -99 | 47 | -2.11 |
| BTC Hourly | lstm | LSTM | 897 | 383 | 514 | 42.70% | 39.17% | 42.29% | 7.30 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 897 | 377 | 520 | 42.03% | 42.50% | 41.88% | 7.97 pp | -143 | 47 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 720 | 348 | 372 | 48.33% | 46.67% | 48.12% | 1.67 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 720 | 344 | 376 | 47.78% | 46.25% | 50.21% | 2.22 pp | -32 | 43 | -0.74 |
| BTC Daily | nn | NN | 720 | 334 | 386 | 46.39% | 44.17% | 47.92% | 3.61 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 720 | 311 | 409 | 43.19% | 37.50% | 41.88% | 6.81 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 720 | 307 | 413 | 42.64% | 40.42% | 43.54% | 7.36 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 730 | 290 | 440 | 39.73% | 36.25% | 38.75% | 10.27 pp | -150 | 43 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 492 | 237 | 255 | 48.17% | 44.58% | 48.12% | 1.83 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 492 | 232 | 260 | 47.15% | 50.00% | 47.71% | 2.85 pp | -28 | 47 | -0.60 |
| BTC Market Hours | transformer | Transformer | 492 | 230 | 262 | 46.75% | 43.33% | 47.29% | 3.25 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 492 | 212 | 280 | 43.09% | 41.25% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 492 | 212 | 280 | 43.09% | 43.33% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 492 | 201 | 291 | 40.85% | 40.00% | 40.83% | 9.15 pp | -90 | 47 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 546 | 253 | 293 | 46.34% | 48.75% | 47.29% | 3.66 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 546 | 251 | 295 | 45.97% | 44.17% | 47.08% | 4.03 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 546 | 250 | 296 | 45.79% | 47.92% | 46.88% | 4.21 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 546 | 227 | 319 | 41.58% | 41.67% | 41.25% | 8.42 pp | -92 | 47 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 546 | 220 | 326 | 40.29% | 37.92% | 40.83% | 9.71 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 546 | 219 | 327 | 40.11% | 40.83% | 39.58% | 9.89 pp | -108 | 47 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| Consolidated Hourly | xgb | XGBoost | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 140 | 66 | 74 | 47.14% | 47.14% | 47.14% | 2.86 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 11 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 140 | 72 | 68 | 51.43% | 51.43% | 51.43% | 1.43 pp | 4 | 11 | 0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 140 | 69 | 71 | 49.29% | 49.29% | 49.29% | 0.71 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 140 | 66 | 74 | 47.14% | 47.14% | 47.14% | 2.86 pp | -8 | 11 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 140 | 64 | 76 | 45.71% | 45.71% | 45.71% | 4.29 pp | -12 | 11 | -1.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 140 | 59 | 81 | 42.14% | 42.14% | 42.14% | 7.86 pp | -22 | 11 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 31 | 9 | 22 | 29.03% | 29.03% | 29.03% | 20.97 pp | -13 | 3 | -4.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
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
