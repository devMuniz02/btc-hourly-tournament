# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T06:51:47.750319+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1256 | 968 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1132 | 767 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 826 | 529 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 828 | 583 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 176 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 529 | 258 | 271 | 48.77% | 46.25% | 48.75% | 1.23 pp | -13 | 50 | -0.26 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| BTC Market Hours | transformer | Transformer | 529 | 253 | 276 | 47.83% | 47.92% | 48.33% | 2.17 pp | -23 | 50 | -0.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 757 | 368 | 389 | 48.61% | 47.92% | 48.75% | 1.39 pp | -21 | 44 | -0.48 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 583 | 277 | 306 | 47.51% | 51.25% | 48.75% | 2.49 pp | -29 | 50 | -0.58 |
| BTC Market Hours | nn | NN | 529 | 250 | 279 | 47.26% | 50.83% | 48.75% | 2.74 pp | -29 | 50 | -0.58 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 583 | 272 | 311 | 46.66% | 46.25% | 47.92% | 3.34 pp | -39 | 50 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 934 | 446 | 488 | 47.75% | 50.00% | 47.29% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 583 | 270 | 313 | 46.31% | 51.25% | 47.08% | 3.69 pp | -43 | 50 | -0.86 |
| BTC Daily | transformer | Transformer | 757 | 357 | 400 | 47.16% | 43.33% | 48.33% | 2.84 pp | -43 | 44 | -0.98 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 934 | 438 | 496 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| BTC Daily | nn | NN | 757 | 351 | 406 | 46.37% | 44.58% | 46.25% | 3.63 pp | -55 | 44 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| BTC Market Hours | lstm | LSTM | 529 | 228 | 301 | 43.10% | 41.67% | 43.96% | 6.90 pp | -73 | 50 | -1.46 |
| BTC Market Hours | rf | RandomForest | 529 | 227 | 302 | 42.91% | 44.17% | 43.54% | 7.09 pp | -75 | 50 | -1.50 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 529 | 218 | 311 | 41.21% | 42.92% | 41.67% | 8.79 pp | -93 | 50 | -1.86 |
| BTC Market Hours Daily | rf | RandomForest | 583 | 242 | 341 | 41.51% | 43.75% | 41.25% | 8.49 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 934 | 416 | 518 | 44.54% | 44.58% | 44.17% | 5.46 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 934 | 414 | 520 | 44.33% | 42.50% | 42.29% | 5.67 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 583 | 236 | 347 | 40.48% | 40.00% | 40.42% | 9.52 pp | -111 | 50 | -2.22 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 583 | 232 | 351 | 39.79% | 40.83% | 38.96% | 10.21 pp | -119 | 50 | -2.38 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 757 | 320 | 437 | 42.27% | 35.42% | 40.21% | 7.73 pp | -117 | 44 | -2.66 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 934 | 399 | 535 | 42.72% | 37.50% | 41.88% | 7.28 pp | -136 | 49 | -2.78 |
| BTC Daily | rf | RandomForest | 757 | 316 | 441 | 41.74% | 37.92% | 41.67% | 8.26 pp | -125 | 44 | -2.84 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 934 | 393 | 541 | 42.08% | 41.25% | 40.83% | 7.92 pp | -148 | 49 | -3.02 |
| BTC Daily | xgb | XGBoost | 767 | 300 | 467 | 39.11% | 35.00% | 36.67% | 10.89 pp | -167 | 44 | -3.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 934 | 446 | 488 | 47.75% | 50.00% | 47.29% | 2.25 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 934 | 438 | 496 | 46.90% | 45.83% | 45.42% | 3.10 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 934 | 416 | 518 | 44.54% | 44.58% | 44.17% | 5.46 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 934 | 414 | 520 | 44.33% | 42.50% | 42.29% | 5.67 pp | -106 | 49 | -2.16 |
| BTC Hourly | lstm | LSTM | 934 | 399 | 535 | 42.72% | 37.50% | 41.88% | 7.28 pp | -136 | 49 | -2.78 |
| BTC Hourly | xgb | XGBoost | 934 | 393 | 541 | 42.08% | 41.25% | 40.83% | 7.92 pp | -148 | 49 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 757 | 368 | 389 | 48.61% | 47.92% | 48.75% | 1.39 pp | -21 | 44 | -0.48 |
| BTC Daily | transformer | Transformer | 757 | 357 | 400 | 47.16% | 43.33% | 48.33% | 2.84 pp | -43 | 44 | -0.98 |
| BTC Daily | nn | NN | 757 | 351 | 406 | 46.37% | 44.58% | 46.25% | 3.63 pp | -55 | 44 | -1.25 |
| BTC Daily | lstm | LSTM | 757 | 320 | 437 | 42.27% | 35.42% | 40.21% | 7.73 pp | -117 | 44 | -2.66 |
| BTC Daily | rf | RandomForest | 757 | 316 | 441 | 41.74% | 37.92% | 41.67% | 8.26 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 767 | 300 | 467 | 39.11% | 35.00% | 36.67% | 10.89 pp | -167 | 44 | -3.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 529 | 258 | 271 | 48.77% | 46.25% | 48.75% | 1.23 pp | -13 | 50 | -0.26 |
| BTC Market Hours | transformer | Transformer | 529 | 253 | 276 | 47.83% | 47.92% | 48.33% | 2.17 pp | -23 | 50 | -0.46 |
| BTC Market Hours | nn | NN | 529 | 250 | 279 | 47.26% | 50.83% | 48.75% | 2.74 pp | -29 | 50 | -0.58 |
| BTC Market Hours | lstm | LSTM | 529 | 228 | 301 | 43.10% | 41.67% | 43.96% | 6.90 pp | -73 | 50 | -1.46 |
| BTC Market Hours | rf | RandomForest | 529 | 227 | 302 | 42.91% | 44.17% | 43.54% | 7.09 pp | -75 | 50 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 529 | 218 | 311 | 41.21% | 42.92% | 41.67% | 8.79 pp | -93 | 50 | -1.86 |

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
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 22 | 29 | 43.14% | 43.14% | 43.14% | 6.86 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
