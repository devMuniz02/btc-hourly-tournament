# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T04:47:12.961523+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1174 | 886 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1049 | 684 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 678 | 446 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 680 | 500 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 446 | 218 | 228 | 48.88% | 45.42% | 48.88% | 1.12 pp | -10 | 44 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 674 | 331 | 343 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| BTC Daily | transformer | Transformer | 674 | 327 | 347 | 48.52% | 46.25% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 446 | 210 | 236 | 47.09% | 48.75% | 47.09% | 2.91 pp | -26 | 44 | -0.59 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 446 | 205 | 241 | 45.96% | 40.83% | 45.96% | 4.04 pp | -36 | 44 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 500 | 231 | 269 | 46.20% | 47.08% | 46.67% | 3.80 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | nn | NN | 500 | 229 | 271 | 45.80% | 43.75% | 46.67% | 4.20 pp | -42 | 44 | -0.95 |
| BTC Hourly | transformer | Transformer | 852 | 402 | 450 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 500 | 226 | 274 | 45.20% | 45.00% | 45.21% | 4.80 pp | -48 | 44 | -1.09 |
| BTC Daily | nn | NN | 674 | 314 | 360 | 46.59% | 42.50% | 48.75% | 3.41 pp | -46 | 41 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 852 | 400 | 452 | 46.95% | 44.17% | 46.67% | 3.05 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 446 | 194 | 252 | 43.50% | 43.75% | 43.50% | 6.50 pp | -58 | 44 | -1.32 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 446 | 191 | 255 | 42.83% | 40.42% | 42.83% | 7.17 pp | -64 | 44 | -1.45 |
| BTC Hourly | nn | NN | 852 | 385 | 467 | 45.19% | 45.00% | 44.58% | 4.81 pp | -82 | 45 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 500 | 208 | 292 | 41.60% | 42.08% | 41.88% | 8.40 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 446 | 180 | 266 | 40.36% | 38.75% | 40.36% | 9.64 pp | -86 | 44 | -1.95 |
| BTC Hourly | rf | RandomForest | 852 | 379 | 473 | 44.48% | 42.92% | 43.75% | 5.52 pp | -94 | 45 | -2.09 |
| BTC Daily | lstm | LSTM | 674 | 294 | 380 | 43.62% | 38.75% | 42.92% | 6.38 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 500 | 200 | 300 | 40.00% | 37.50% | 40.62% | 10.00 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 674 | 289 | 385 | 42.88% | 40.83% | 43.54% | 7.12 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 500 | 196 | 304 | 39.20% | 36.25% | 38.96% | 10.80 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 852 | 364 | 488 | 42.72% | 38.75% | 42.29% | 7.28 pp | -124 | 45 | -2.76 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Hourly | xgb | XGBoost | 852 | 358 | 494 | 42.02% | 40.00% | 42.29% | 7.98 pp | -136 | 45 | -3.02 |
| BTC Daily | xgb | XGBoost | 684 | 271 | 413 | 39.62% | 34.58% | 39.38% | 10.38 pp | -142 | 41 | -3.46 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 852 | 402 | 450 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 852 | 400 | 452 | 46.95% | 44.17% | 46.67% | 3.05 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 852 | 385 | 467 | 45.19% | 45.00% | 44.58% | 4.81 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 852 | 379 | 473 | 44.48% | 42.92% | 43.75% | 5.52 pp | -94 | 45 | -2.09 |
| BTC Hourly | lstm | LSTM | 852 | 364 | 488 | 42.72% | 38.75% | 42.29% | 7.28 pp | -124 | 45 | -2.76 |
| BTC Hourly | xgb | XGBoost | 852 | 358 | 494 | 42.02% | 40.00% | 42.29% | 7.98 pp | -136 | 45 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 674 | 331 | 343 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| BTC Daily | transformer | Transformer | 674 | 327 | 347 | 48.52% | 46.25% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| BTC Daily | nn | NN | 674 | 314 | 360 | 46.59% | 42.50% | 48.75% | 3.41 pp | -46 | 41 | -1.12 |
| BTC Daily | lstm | LSTM | 674 | 294 | 380 | 43.62% | 38.75% | 42.92% | 6.38 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 674 | 289 | 385 | 42.88% | 40.83% | 43.54% | 7.12 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 684 | 271 | 413 | 39.62% | 34.58% | 39.38% | 10.38 pp | -142 | 41 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 446 | 218 | 228 | 48.88% | 45.42% | 48.88% | 1.12 pp | -10 | 44 | -0.23 |
| BTC Market Hours | nn | NN | 446 | 210 | 236 | 47.09% | 48.75% | 47.09% | 2.91 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 446 | 205 | 241 | 45.96% | 40.83% | 45.96% | 4.04 pp | -36 | 44 | -0.82 |
| BTC Market Hours | rf | RandomForest | 446 | 194 | 252 | 43.50% | 43.75% | 43.50% | 6.50 pp | -58 | 44 | -1.32 |
| BTC Market Hours | lstm | LSTM | 446 | 191 | 255 | 42.83% | 40.42% | 42.83% | 7.17 pp | -64 | 44 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 446 | 180 | 266 | 40.36% | 38.75% | 40.36% | 9.64 pp | -86 | 44 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 500 | 231 | 269 | 46.20% | 47.08% | 46.67% | 3.80 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | nn | NN | 500 | 229 | 271 | 45.80% | 43.75% | 46.67% | 4.20 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 500 | 226 | 274 | 45.20% | 45.00% | 45.21% | 4.80 pp | -48 | 44 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 500 | 208 | 292 | 41.60% | 42.08% | 41.88% | 8.40 pp | -84 | 44 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 500 | 200 | 300 | 40.00% | 37.50% | 40.62% | 10.00 pp | -100 | 44 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 500 | 196 | 304 | 39.20% | 36.25% | 38.96% | 10.80 pp | -108 | 44 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
