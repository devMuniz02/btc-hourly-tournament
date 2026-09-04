# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T07:52:25.233136+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1224 | 936 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1100 | 735 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 768 | 497 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 770 | 551 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 145 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T18:00:00+00:00 | 146 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 497 | 241 | 256 | 48.49% | 45.00% | 48.33% | 1.51 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 497 | 235 | 262 | 47.28% | 50.42% | 47.92% | 2.72 pp | -27 | 48 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 725 | 349 | 376 | 48.14% | 45.83% | 47.71% | 1.86 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 497 | 233 | 264 | 46.88% | 44.17% | 47.71% | 3.12 pp | -31 | 48 | -0.65 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 725 | 346 | 379 | 47.72% | 46.67% | 49.79% | 2.28 pp | -33 | 43 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 551 | 257 | 294 | 46.64% | 49.17% | 47.71% | 3.36 pp | -37 | 48 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 902 | 432 | 470 | 47.89% | 51.25% | 48.33% | 2.11 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 551 | 254 | 297 | 46.10% | 49.17% | 46.88% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 551 | 254 | 297 | 46.10% | 45.00% | 47.50% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Hourly | transformer | Transformer | 902 | 427 | 475 | 47.34% | 47.92% | 46.88% | 2.66 pp | -48 | 48 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| BTC Daily | nn | NN | 725 | 336 | 389 | 46.34% | 44.58% | 47.50% | 3.66 pp | -53 | 43 | -1.23 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| BTC Market Hours | lstm | LSTM | 497 | 214 | 283 | 43.06% | 40.83% | 43.12% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 497 | 214 | 283 | 43.06% | 43.75% | 43.33% | 6.94 pp | -69 | 48 | -1.44 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 497 | 204 | 293 | 41.05% | 41.25% | 41.25% | 8.95 pp | -89 | 48 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 551 | 229 | 322 | 41.56% | 42.50% | 41.04% | 8.44 pp | -93 | 48 | -1.94 |
| BTC Hourly | nn | NN | 902 | 402 | 500 | 44.57% | 44.58% | 42.29% | 5.43 pp | -98 | 48 | -2.04 |
| BTC Hourly | rf | RandomForest | 902 | 401 | 501 | 44.46% | 44.17% | 44.17% | 5.54 pp | -100 | 48 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 551 | 222 | 329 | 40.29% | 38.75% | 40.62% | 9.71 pp | -107 | 48 | -2.23 |
| Consolidated Hourly | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 551 | 220 | 331 | 39.93% | 41.25% | 39.17% | 10.07 pp | -111 | 48 | -2.31 |
| BTC Daily | lstm | LSTM | 725 | 312 | 413 | 43.03% | 37.08% | 41.46% | 6.97 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 725 | 310 | 415 | 42.76% | 40.83% | 43.54% | 7.24 pp | -105 | 43 | -2.44 |
| BTC Hourly | lstm | LSTM | 902 | 387 | 515 | 42.90% | 40.00% | 42.29% | 7.10 pp | -128 | 48 | -2.67 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | xgb | XGBoost | 902 | 379 | 523 | 42.02% | 41.67% | 41.46% | 7.98 pp | -144 | 48 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 735 | 292 | 443 | 39.73% | 37.08% | 38.75% | 10.27 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 902 | 432 | 470 | 47.89% | 51.25% | 48.33% | 2.11 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 902 | 427 | 475 | 47.34% | 47.92% | 46.88% | 2.66 pp | -48 | 48 | -1.00 |
| BTC Hourly | nn | NN | 902 | 402 | 500 | 44.57% | 44.58% | 42.29% | 5.43 pp | -98 | 48 | -2.04 |
| BTC Hourly | rf | RandomForest | 902 | 401 | 501 | 44.46% | 44.17% | 44.17% | 5.54 pp | -100 | 48 | -2.08 |
| BTC Hourly | lstm | LSTM | 902 | 387 | 515 | 42.90% | 40.00% | 42.29% | 7.10 pp | -128 | 48 | -2.67 |
| BTC Hourly | xgb | XGBoost | 902 | 379 | 523 | 42.02% | 41.67% | 41.46% | 7.98 pp | -144 | 48 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 725 | 349 | 376 | 48.14% | 45.83% | 47.71% | 1.86 pp | -27 | 43 | -0.63 |
| BTC Daily | transformer | Transformer | 725 | 346 | 379 | 47.72% | 46.67% | 49.79% | 2.28 pp | -33 | 43 | -0.77 |
| BTC Daily | nn | NN | 725 | 336 | 389 | 46.34% | 44.58% | 47.50% | 3.66 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 725 | 312 | 413 | 43.03% | 37.08% | 41.46% | 6.97 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 725 | 310 | 415 | 42.76% | 40.83% | 43.54% | 7.24 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 735 | 292 | 443 | 39.73% | 37.08% | 38.75% | 10.27 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 497 | 241 | 256 | 48.49% | 45.00% | 48.33% | 1.51 pp | -15 | 48 | -0.31 |
| BTC Market Hours | nn | NN | 497 | 235 | 262 | 47.28% | 50.42% | 47.92% | 2.72 pp | -27 | 48 | -0.56 |
| BTC Market Hours | transformer | Transformer | 497 | 233 | 264 | 46.88% | 44.17% | 47.71% | 3.12 pp | -31 | 48 | -0.65 |
| BTC Market Hours | lstm | LSTM | 497 | 214 | 283 | 43.06% | 40.83% | 43.12% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | rf | RandomForest | 497 | 214 | 283 | 43.06% | 43.75% | 43.33% | 6.94 pp | -69 | 48 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 497 | 204 | 293 | 41.05% | 41.25% | 41.25% | 8.95 pp | -89 | 48 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 551 | 257 | 294 | 46.64% | 49.17% | 47.71% | 3.36 pp | -37 | 48 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 551 | 254 | 297 | 46.10% | 49.17% | 46.88% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | nn | NN | 551 | 254 | 297 | 46.10% | 45.00% | 47.50% | 3.90 pp | -43 | 48 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 551 | 229 | 322 | 41.56% | 42.50% | 41.04% | 8.44 pp | -93 | 48 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 551 | 222 | 329 | 40.29% | 38.75% | 40.62% | 9.71 pp | -107 | 48 | -2.23 |
| BTC Market Hours Daily | xgb | XGBoost | 551 | 220 | 331 | 39.93% | 41.25% | 39.17% | 10.07 pp | -111 | 48 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 74 | 71 | 51.03% | 51.03% | 51.03% | 1.03 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 71 | 74 | 48.97% | 48.97% | 48.97% | 1.03 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 66 | 79 | 45.52% | 45.52% | 45.52% | 4.48 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 65 | 80 | 44.83% | 44.83% | 44.83% | 5.17 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 60 | 85 | 41.38% | 41.38% | 41.38% | 8.62 pp | -25 | 11 | -2.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | nn | NN | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
