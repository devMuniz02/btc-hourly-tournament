# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T19:45:49.021874+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1167 | 879 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1043 | 678 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 18:00:00+00:00 | 666 | 440 | 225 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 18:00:00+00:00 | 668 | 494 | 172 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 92 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 92 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 92 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 93 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | xgb | XGBoost | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 92 | 47 | 45 | 51.09% | 51.09% | 51.09% | 1.09 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 92 | 47 | 45 | 51.09% | 51.09% | 51.09% | 1.09 pp | 2 | 9 | 0.22 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | nn | NN | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 9 | -0.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 9 | -0.22 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 440 | 215 | 225 | 48.86% | 44.58% | 48.86% | 1.14 pp | -10 | 43 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 668 | 326 | 342 | 48.80% | 46.67% | 49.58% | 1.20 pp | -16 | 41 | -0.39 |
| Consolidated Hourly | lstm | LSTM | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 9 | -0.44 |
| BTC Daily | transformer | Transformer | 668 | 323 | 345 | 48.35% | 45.83% | 49.79% | 1.65 pp | -22 | 41 | -0.54 |
| BTC Market Hours | nn | NN | 440 | 208 | 232 | 47.27% | 49.17% | 47.27% | 2.73 pp | -24 | 43 | -0.56 |
| BTC Market Hours | transformer | Transformer | 440 | 202 | 238 | 45.91% | 40.83% | 45.91% | 4.09 pp | -36 | 43 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 494 | 228 | 266 | 46.15% | 47.08% | 46.46% | 3.85 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | nn | NN | 494 | 226 | 268 | 45.75% | 43.75% | 46.25% | 4.25 pp | -42 | 43 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 668 | 313 | 355 | 46.86% | 43.33% | 49.38% | 3.14 pp | -42 | 41 | -1.02 |
| BTC Hourly | transformer | Transformer | 845 | 399 | 446 | 47.22% | 47.92% | 47.08% | 2.78 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | transformer | Transformer | 494 | 224 | 270 | 45.34% | 45.00% | 45.21% | 4.66 pp | -46 | 43 | -1.07 |
| Consolidated Hourly | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 9 | -1.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 9 | -1.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 845 | 397 | 448 | 46.98% | 44.17% | 46.88% | 3.02 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 440 | 189 | 251 | 42.95% | 43.33% | 42.95% | 7.05 pp | -62 | 43 | -1.44 |
| BTC Market Hours | lstm | LSTM | 440 | 188 | 252 | 42.73% | 41.67% | 42.73% | 7.27 pp | -64 | 43 | -1.49 |
| BTC Hourly | nn | NN | 845 | 381 | 464 | 45.09% | 44.17% | 44.58% | 4.91 pp | -83 | 45 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 440 | 177 | 263 | 40.23% | 38.75% | 40.23% | 9.77 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 494 | 203 | 291 | 41.09% | 41.25% | 41.25% | 8.91 pp | -88 | 43 | -2.05 |
| BTC Daily | lstm | LSTM | 668 | 292 | 376 | 43.71% | 38.75% | 42.92% | 6.29 pp | -84 | 41 | -2.05 |
| BTC Hourly | rf | RandomForest | 845 | 376 | 469 | 44.50% | 43.75% | 43.96% | 5.50 pp | -93 | 45 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 494 | 197 | 297 | 39.88% | 37.50% | 40.62% | 10.12 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 668 | 285 | 383 | 42.66% | 40.42% | 43.75% | 7.34 pp | -98 | 41 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 494 | 193 | 301 | 39.07% | 36.25% | 39.17% | 10.93 pp | -108 | 43 | -2.51 |
| BTC Hourly | lstm | LSTM | 845 | 362 | 483 | 42.84% | 40.00% | 42.29% | 7.16 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 845 | 356 | 489 | 42.13% | 40.00% | 42.50% | 7.87 pp | -133 | 45 | -2.96 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 678 | 269 | 409 | 39.68% | 34.17% | 39.79% | 10.32 pp | -140 | 41 | -3.41 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 845 | 399 | 446 | 47.22% | 47.92% | 47.08% | 2.78 pp | -47 | 45 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 845 | 397 | 448 | 46.98% | 44.17% | 46.88% | 3.02 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 845 | 381 | 464 | 45.09% | 44.17% | 44.58% | 4.91 pp | -83 | 45 | -1.84 |
| BTC Hourly | rf | RandomForest | 845 | 376 | 469 | 44.50% | 43.75% | 43.96% | 5.50 pp | -93 | 45 | -2.07 |
| BTC Hourly | lstm | LSTM | 845 | 362 | 483 | 42.84% | 40.00% | 42.29% | 7.16 pp | -121 | 45 | -2.69 |
| BTC Hourly | xgb | XGBoost | 845 | 356 | 489 | 42.13% | 40.00% | 42.50% | 7.87 pp | -133 | 45 | -2.96 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 668 | 326 | 342 | 48.80% | 46.67% | 49.58% | 1.20 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 668 | 323 | 345 | 48.35% | 45.83% | 49.79% | 1.65 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 668 | 313 | 355 | 46.86% | 43.33% | 49.38% | 3.14 pp | -42 | 41 | -1.02 |
| BTC Daily | lstm | LSTM | 668 | 292 | 376 | 43.71% | 38.75% | 42.92% | 6.29 pp | -84 | 41 | -2.05 |
| BTC Daily | rf | RandomForest | 668 | 285 | 383 | 42.66% | 40.42% | 43.75% | 7.34 pp | -98 | 41 | -2.39 |
| BTC Daily | xgb | XGBoost | 678 | 269 | 409 | 39.68% | 34.17% | 39.79% | 10.32 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 440 | 215 | 225 | 48.86% | 44.58% | 48.86% | 1.14 pp | -10 | 43 | -0.23 |
| BTC Market Hours | nn | NN | 440 | 208 | 232 | 47.27% | 49.17% | 47.27% | 2.73 pp | -24 | 43 | -0.56 |
| BTC Market Hours | transformer | Transformer | 440 | 202 | 238 | 45.91% | 40.83% | 45.91% | 4.09 pp | -36 | 43 | -0.84 |
| BTC Market Hours | rf | RandomForest | 440 | 189 | 251 | 42.95% | 43.33% | 42.95% | 7.05 pp | -62 | 43 | -1.44 |
| BTC Market Hours | lstm | LSTM | 440 | 188 | 252 | 42.73% | 41.67% | 42.73% | 7.27 pp | -64 | 43 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 440 | 177 | 263 | 40.23% | 38.75% | 40.23% | 9.77 pp | -86 | 43 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 494 | 228 | 266 | 46.15% | 47.08% | 46.46% | 3.85 pp | -38 | 43 | -0.88 |
| BTC Market Hours Daily | nn | NN | 494 | 226 | 268 | 45.75% | 43.75% | 46.25% | 4.25 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 494 | 224 | 270 | 45.34% | 45.00% | 45.21% | 4.66 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 494 | 203 | 291 | 41.09% | 41.25% | 41.25% | 8.91 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 494 | 197 | 297 | 39.88% | 37.50% | 40.62% | 10.12 pp | -100 | 43 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 494 | 193 | 301 | 39.07% | 36.25% | 39.17% | 10.93 pp | -108 | 43 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | xgb | XGBoost | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 92 | 47 | 45 | 51.09% | 51.09% | 51.09% | 1.09 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | nn | NN | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 9 | -0.22 |
| Consolidated Hourly | lstm | LSTM | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 9 | -1.11 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 92 | 47 | 45 | 51.09% | 51.09% | 51.09% | 1.09 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 9 | -0.22 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 92 | 41 | 51 | 44.57% | 44.57% | 44.57% | 5.43 pp | -10 | 9 | -1.11 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
