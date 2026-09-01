# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T06:29:51.142691+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1175 | 887 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1051 | 686 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 680 | 448 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 682 | 502 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 100 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 100 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 100 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 101 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | xgb | XGBoost | 100 | 50 | 50 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 100 | 50 | 50 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 448 | 218 | 230 | 48.66% | 45.00% | 48.66% | 1.34 pp | -12 | 44 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 676 | 332 | 344 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| Consolidated Hourly | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 9 | -0.44 |
| BTC Daily | transformer | Transformer | 676 | 328 | 348 | 48.52% | 46.67% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| BTC Market Hours | nn | NN | 448 | 210 | 238 | 46.88% | 47.92% | 46.88% | 3.12 pp | -28 | 44 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 9 | -0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 9 | -0.67 |
| BTC Market Hours | transformer | Transformer | 448 | 206 | 242 | 45.98% | 40.42% | 45.98% | 4.02 pp | -36 | 44 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 502 | 231 | 271 | 46.02% | 46.67% | 46.46% | 3.98 pp | -40 | 44 | -0.91 |
| BTC Market Hours Daily | nn | NN | 502 | 229 | 273 | 45.62% | 42.92% | 46.25% | 4.38 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 502 | 228 | 274 | 45.42% | 45.83% | 45.62% | 4.58 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 853 | 402 | 451 | 47.13% | 47.08% | 46.88% | 2.87 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 676 | 316 | 360 | 46.75% | 43.33% | 48.96% | 3.25 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 853 | 401 | 452 | 47.01% | 44.58% | 46.88% | 2.99 pp | -51 | 46 | -1.11 |
| BTC Market Hours | rf | RandomForest | 448 | 194 | 254 | 43.30% | 42.92% | 43.30% | 6.70 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 448 | 191 | 257 | 42.63% | 40.00% | 42.63% | 7.37 pp | -66 | 44 | -1.50 |
| BTC Hourly | nn | NN | 853 | 385 | 468 | 45.13% | 45.00% | 44.58% | 4.87 pp | -83 | 46 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 502 | 209 | 293 | 41.63% | 42.08% | 41.67% | 8.37 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 448 | 180 | 268 | 40.18% | 37.92% | 40.18% | 9.82 pp | -88 | 44 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 853 | 379 | 474 | 44.43% | 42.92% | 43.75% | 5.57 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 676 | 295 | 381 | 43.64% | 38.75% | 42.92% | 6.36 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 502 | 201 | 301 | 40.04% | 37.50% | 40.83% | 9.96 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 676 | 291 | 385 | 43.05% | 41.25% | 43.75% | 6.95 pp | -94 | 41 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 502 | 197 | 305 | 39.24% | 36.25% | 38.96% | 10.76 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 853 | 364 | 489 | 42.67% | 38.33% | 42.08% | 7.33 pp | -125 | 46 | -2.72 |
| BTC Hourly | xgb | XGBoost | 853 | 358 | 495 | 41.97% | 39.58% | 42.08% | 8.03 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 686 | 273 | 413 | 39.80% | 35.42% | 39.58% | 10.20 pp | -140 | 41 | -3.41 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 853 | 402 | 451 | 47.13% | 47.08% | 46.88% | 2.87 pp | -49 | 46 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 853 | 401 | 452 | 47.01% | 44.58% | 46.88% | 2.99 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 853 | 385 | 468 | 45.13% | 45.00% | 44.58% | 4.87 pp | -83 | 46 | -1.80 |
| BTC Hourly | rf | RandomForest | 853 | 379 | 474 | 44.43% | 42.92% | 43.75% | 5.57 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 853 | 364 | 489 | 42.67% | 38.33% | 42.08% | 7.33 pp | -125 | 46 | -2.72 |
| BTC Hourly | xgb | XGBoost | 853 | 358 | 495 | 41.97% | 39.58% | 42.08% | 8.03 pp | -137 | 46 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 676 | 332 | 344 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| BTC Daily | transformer | Transformer | 676 | 328 | 348 | 48.52% | 46.67% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| BTC Daily | nn | NN | 676 | 316 | 360 | 46.75% | 43.33% | 48.96% | 3.25 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 676 | 295 | 381 | 43.64% | 38.75% | 42.92% | 6.36 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 676 | 291 | 385 | 43.05% | 41.25% | 43.75% | 6.95 pp | -94 | 41 | -2.29 |
| BTC Daily | xgb | XGBoost | 686 | 273 | 413 | 39.80% | 35.42% | 39.58% | 10.20 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 448 | 218 | 230 | 48.66% | 45.00% | 48.66% | 1.34 pp | -12 | 44 | -0.27 |
| BTC Market Hours | nn | NN | 448 | 210 | 238 | 46.88% | 47.92% | 46.88% | 3.12 pp | -28 | 44 | -0.64 |
| BTC Market Hours | transformer | Transformer | 448 | 206 | 242 | 45.98% | 40.42% | 45.98% | 4.02 pp | -36 | 44 | -0.82 |
| BTC Market Hours | rf | RandomForest | 448 | 194 | 254 | 43.30% | 42.92% | 43.30% | 6.70 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 448 | 191 | 257 | 42.63% | 40.00% | 42.63% | 7.37 pp | -66 | 44 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 448 | 180 | 268 | 40.18% | 37.92% | 40.18% | 9.82 pp | -88 | 44 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 502 | 231 | 271 | 46.02% | 46.67% | 46.46% | 3.98 pp | -40 | 44 | -0.91 |
| BTC Market Hours Daily | nn | NN | 502 | 229 | 273 | 45.62% | 42.92% | 46.25% | 4.38 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 502 | 228 | 274 | 45.42% | 45.83% | 45.62% | 4.58 pp | -46 | 44 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 502 | 209 | 293 | 41.63% | 42.08% | 41.67% | 8.37 pp | -84 | 44 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 502 | 201 | 301 | 40.04% | 37.50% | 40.83% | 9.96 pp | -100 | 44 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 502 | 197 | 305 | 39.24% | 36.25% | 38.96% | 10.76 pp | -108 | 44 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | rf | RandomForest | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | xgb | XGBoost | 100 | 50 | 50 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | lstm | LSTM | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 100 | 52 | 48 | 52.00% | 52.00% | 52.00% | 2.00 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 100 | 50 | 50 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 9 | -0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
