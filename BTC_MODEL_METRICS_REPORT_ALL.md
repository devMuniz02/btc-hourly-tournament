# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T07:30:57.033494+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1176 | 888 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1051 | 686 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 680 | 448 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 682 | 502 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 448 | 218 | 230 | 48.66% | 45.00% | 48.66% | 1.34 pp | -12 | 44 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 676 | 331 | 345 | 48.96% | 47.92% | 49.79% | 1.04 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 676 | 327 | 349 | 48.37% | 46.25% | 49.38% | 1.63 pp | -22 | 41 | -0.54 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 448 | 210 | 238 | 46.88% | 47.92% | 46.88% | 3.12 pp | -28 | 44 | -0.64 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 448 | 206 | 242 | 45.98% | 40.42% | 45.98% | 4.02 pp | -36 | 44 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 502 | 231 | 271 | 46.02% | 46.67% | 46.46% | 3.98 pp | -40 | 44 | -0.91 |
| BTC Market Hours Daily | nn | NN | 502 | 229 | 273 | 45.62% | 42.92% | 46.25% | 4.38 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 502 | 228 | 274 | 45.42% | 45.83% | 45.62% | 4.58 pp | -46 | 44 | -1.05 |
| BTC Daily | nn | NN | 676 | 316 | 360 | 46.75% | 43.33% | 48.96% | 3.25 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 854 | 402 | 452 | 47.07% | 45.00% | 46.88% | 2.93 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 854 | 402 | 452 | 47.07% | 47.08% | 46.88% | 2.93 pp | -50 | 46 | -1.09 |
| BTC Market Hours | rf | RandomForest | 448 | 194 | 254 | 43.30% | 42.92% | 43.30% | 6.70 pp | -60 | 44 | -1.36 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 448 | 191 | 257 | 42.63% | 40.00% | 42.63% | 7.37 pp | -66 | 44 | -1.50 |
| BTC Hourly | nn | NN | 854 | 385 | 469 | 45.08% | 45.00% | 44.38% | 4.92 pp | -84 | 46 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 502 | 209 | 293 | 41.63% | 42.08% | 41.67% | 8.37 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 448 | 180 | 268 | 40.18% | 37.92% | 40.18% | 9.82 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 854 | 380 | 474 | 44.50% | 43.33% | 43.96% | 5.50 pp | -94 | 46 | -2.04 |
| BTC Daily | lstm | LSTM | 676 | 295 | 381 | 43.64% | 38.75% | 42.92% | 6.36 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 502 | 201 | 301 | 40.04% | 37.50% | 40.83% | 9.96 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 676 | 290 | 386 | 42.90% | 40.83% | 43.54% | 7.10 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 502 | 197 | 305 | 39.24% | 36.25% | 38.96% | 10.76 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 854 | 364 | 490 | 42.62% | 38.33% | 42.08% | 7.38 pp | -126 | 46 | -2.74 |
| BTC Hourly | xgb | XGBoost | 854 | 358 | 496 | 41.92% | 39.58% | 42.08% | 8.08 pp | -138 | 46 | -3.00 |
| BTC Daily | xgb | XGBoost | 686 | 272 | 414 | 39.65% | 35.00% | 39.38% | 10.35 pp | -142 | 41 | -3.46 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 854 | 402 | 452 | 47.07% | 45.00% | 46.88% | 2.93 pp | -50 | 46 | -1.09 |
| BTC Hourly | transformer | Transformer | 854 | 402 | 452 | 47.07% | 47.08% | 46.88% | 2.93 pp | -50 | 46 | -1.09 |
| BTC Hourly | nn | NN | 854 | 385 | 469 | 45.08% | 45.00% | 44.38% | 4.92 pp | -84 | 46 | -1.83 |
| BTC Hourly | rf | RandomForest | 854 | 380 | 474 | 44.50% | 43.33% | 43.96% | 5.50 pp | -94 | 46 | -2.04 |
| BTC Hourly | lstm | LSTM | 854 | 364 | 490 | 42.62% | 38.33% | 42.08% | 7.38 pp | -126 | 46 | -2.74 |
| BTC Hourly | xgb | XGBoost | 854 | 358 | 496 | 41.92% | 39.58% | 42.08% | 8.08 pp | -138 | 46 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 676 | 331 | 345 | 48.96% | 47.92% | 49.79% | 1.04 pp | -14 | 41 | -0.34 |
| BTC Daily | transformer | Transformer | 676 | 327 | 349 | 48.37% | 46.25% | 49.38% | 1.63 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 676 | 316 | 360 | 46.75% | 43.33% | 48.96% | 3.25 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 676 | 295 | 381 | 43.64% | 38.75% | 42.92% | 6.36 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 676 | 290 | 386 | 42.90% | 40.83% | 43.54% | 7.10 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 686 | 272 | 414 | 39.65% | 35.00% | 39.38% | 10.35 pp | -142 | 41 | -3.46 |

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
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
