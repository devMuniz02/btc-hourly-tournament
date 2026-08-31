# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T15:50:02.208668+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1164 | 876 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1040 | 675 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 14:00:00+00:00 | 659 | 437 | 221 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 14:00:00+00:00 | 661 | 491 | 168 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 89 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T14:00:00+00:00 | 90 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 437 | 215 | 222 | 49.20% | 45.42% | 49.20% | 0.80 pp | -7 | 43 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 665 | 326 | 339 | 49.02% | 47.50% | 50.21% | 0.98 pp | -13 | 41 | -0.32 |
| Consolidated Hourly | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| BTC Daily | transformer | Transformer | 665 | 320 | 345 | 48.12% | 45.00% | 49.38% | 1.88 pp | -25 | 41 | -0.61 |
| BTC Market Hours | nn | NN | 437 | 205 | 232 | 46.91% | 48.75% | 46.91% | 3.09 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 491 | 228 | 263 | 46.44% | 47.08% | 46.67% | 3.56 pp | -35 | 43 | -0.81 |
| BTC Market Hours | transformer | Transformer | 437 | 201 | 236 | 46.00% | 41.25% | 46.00% | 4.00 pp | -35 | 43 | -0.81 |
| BTC Daily | nn | NN | 665 | 313 | 352 | 47.07% | 43.75% | 49.79% | 2.93 pp | -39 | 41 | -0.95 |
| BTC Hourly | transformer | Transformer | 842 | 399 | 443 | 47.39% | 47.92% | 47.08% | 2.61 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | nn | NN | 491 | 224 | 267 | 45.62% | 43.75% | 46.04% | 4.38 pp | -43 | 43 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 491 | 223 | 268 | 45.42% | 45.42% | 45.42% | 4.58 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 842 | 395 | 447 | 46.91% | 43.75% | 46.67% | 3.09 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 437 | 188 | 249 | 43.02% | 42.08% | 43.02% | 6.98 pp | -61 | 43 | -1.42 |
| BTC Market Hours | rf | RandomForest | 437 | 188 | 249 | 43.02% | 43.33% | 43.02% | 6.98 pp | -61 | 43 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | nn | NN | 842 | 380 | 462 | 45.13% | 43.75% | 44.58% | 4.87 pp | -82 | 45 | -1.82 |
| BTC Daily | lstm | LSTM | 665 | 292 | 373 | 43.91% | 39.58% | 43.33% | 6.09 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 491 | 203 | 288 | 41.34% | 41.67% | 41.46% | 8.66 pp | -85 | 43 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 437 | 176 | 261 | 40.27% | 38.75% | 40.27% | 9.73 pp | -85 | 43 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 842 | 375 | 467 | 44.54% | 43.33% | 43.96% | 5.46 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 491 | 197 | 294 | 40.12% | 38.75% | 40.62% | 9.88 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 665 | 285 | 380 | 42.86% | 41.25% | 43.96% | 7.14 pp | -95 | 41 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 491 | 192 | 299 | 39.10% | 35.83% | 39.38% | 10.90 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 842 | 361 | 481 | 42.87% | 39.58% | 42.29% | 7.13 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 842 | 355 | 487 | 42.16% | 39.58% | 42.50% | 7.84 pp | -132 | 45 | -2.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 675 | 269 | 406 | 39.85% | 34.58% | 40.00% | 10.15 pp | -137 | 41 | -3.34 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 842 | 399 | 443 | 47.39% | 47.92% | 47.08% | 2.61 pp | -44 | 45 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 842 | 395 | 447 | 46.91% | 43.75% | 46.67% | 3.09 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 842 | 380 | 462 | 45.13% | 43.75% | 44.58% | 4.87 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 842 | 375 | 467 | 44.54% | 43.33% | 43.96% | 5.46 pp | -92 | 45 | -2.04 |
| BTC Hourly | lstm | LSTM | 842 | 361 | 481 | 42.87% | 39.58% | 42.29% | 7.13 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 842 | 355 | 487 | 42.16% | 39.58% | 42.50% | 7.84 pp | -132 | 45 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 665 | 326 | 339 | 49.02% | 47.50% | 50.21% | 0.98 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 665 | 320 | 345 | 48.12% | 45.00% | 49.38% | 1.88 pp | -25 | 41 | -0.61 |
| BTC Daily | nn | NN | 665 | 313 | 352 | 47.07% | 43.75% | 49.79% | 2.93 pp | -39 | 41 | -0.95 |
| BTC Daily | lstm | LSTM | 665 | 292 | 373 | 43.91% | 39.58% | 43.33% | 6.09 pp | -81 | 41 | -1.98 |
| BTC Daily | rf | RandomForest | 665 | 285 | 380 | 42.86% | 41.25% | 43.96% | 7.14 pp | -95 | 41 | -2.32 |
| BTC Daily | xgb | XGBoost | 675 | 269 | 406 | 39.85% | 34.58% | 40.00% | 10.15 pp | -137 | 41 | -3.34 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 437 | 215 | 222 | 49.20% | 45.42% | 49.20% | 0.80 pp | -7 | 43 | -0.16 |
| BTC Market Hours | nn | NN | 437 | 205 | 232 | 46.91% | 48.75% | 46.91% | 3.09 pp | -27 | 43 | -0.63 |
| BTC Market Hours | transformer | Transformer | 437 | 201 | 236 | 46.00% | 41.25% | 46.00% | 4.00 pp | -35 | 43 | -0.81 |
| BTC Market Hours | lstm | LSTM | 437 | 188 | 249 | 43.02% | 42.08% | 43.02% | 6.98 pp | -61 | 43 | -1.42 |
| BTC Market Hours | rf | RandomForest | 437 | 188 | 249 | 43.02% | 43.33% | 43.02% | 6.98 pp | -61 | 43 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 437 | 176 | 261 | 40.27% | 38.75% | 40.27% | 9.73 pp | -85 | 43 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 491 | 228 | 263 | 46.44% | 47.08% | 46.67% | 3.56 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | nn | NN | 491 | 224 | 267 | 45.62% | 43.75% | 46.04% | 4.38 pp | -43 | 43 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 491 | 223 | 268 | 45.42% | 45.42% | 45.42% | 4.58 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 491 | 203 | 288 | 41.34% | 41.67% | 41.46% | 8.66 pp | -85 | 43 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 491 | 197 | 294 | 40.12% | 38.75% | 40.62% | 9.88 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 491 | 192 | 299 | 39.10% | 35.83% | 39.38% | 10.90 pp | -107 | 43 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 42 | 47 | 47.19% | 47.19% | 47.19% | 2.81 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
