# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T18:33:09.931722+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1231 | 943 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1107 | 742 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 781 | 504 | 276 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 17:00:00+00:00 | 783 | 558 | 223 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 152 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 152 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 152 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T21:00:00+00:00 | 153 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 152 | 75 | 77 | 49.34% | 49.34% | 49.34% | 0.66 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 152 | 75 | 77 | 49.34% | 49.34% | 49.34% | 0.66 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 504 | 242 | 262 | 48.02% | 44.58% | 47.92% | 1.98 pp | -20 | 48 | -0.42 |
| BTC Market Hours | transformer | Transformer | 504 | 240 | 264 | 47.62% | 46.25% | 48.12% | 2.38 pp | -24 | 48 | -0.50 |
| BTC Market Hours | nn | NN | 504 | 239 | 265 | 47.42% | 50.83% | 48.33% | 2.58 pp | -26 | 48 | -0.54 |
| BTC Daily | mlp_sklearn | MLPClassifier | 732 | 354 | 378 | 48.36% | 47.08% | 48.12% | 1.64 pp | -24 | 43 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 558 | 263 | 295 | 47.13% | 50.00% | 47.92% | 2.87 pp | -32 | 48 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 11 | -0.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 11 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 909 | 436 | 473 | 47.96% | 51.25% | 48.33% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Daily | transformer | Transformer | 732 | 349 | 383 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | nn | NN | 558 | 259 | 299 | 46.42% | 45.83% | 47.71% | 3.58 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 558 | 258 | 300 | 46.24% | 49.58% | 46.67% | 3.76 pp | -42 | 48 | -0.88 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Hourly | transformer | Transformer | 909 | 429 | 480 | 47.19% | 47.50% | 46.67% | 2.81 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 732 | 339 | 393 | 46.31% | 45.00% | 47.08% | 3.69 pp | -54 | 43 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 504 | 217 | 287 | 43.06% | 41.25% | 43.12% | 6.94 pp | -70 | 48 | -1.46 |
| BTC Market Hours | rf | RandomForest | 504 | 216 | 288 | 42.86% | 44.17% | 43.12% | 7.14 pp | -72 | 48 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | nn | NN | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 11 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 504 | 208 | 296 | 41.27% | 42.50% | 41.88% | 8.73 pp | -88 | 48 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 3 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 558 | 231 | 327 | 41.40% | 42.08% | 40.42% | 8.60 pp | -96 | 48 | -2.00 |
| BTC Hourly | nn | NN | 909 | 404 | 505 | 44.44% | 43.75% | 42.08% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 909 | 404 | 505 | 44.44% | 44.17% | 43.96% | 5.56 pp | -101 | 48 | -2.10 |
| Consolidated Hourly | transformer | Transformer | 152 | 64 | 88 | 42.11% | 42.11% | 42.11% | 7.89 pp | -24 | 11 | -2.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 152 | 64 | 88 | 42.11% | 42.11% | 42.11% | 7.89 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 558 | 225 | 333 | 40.32% | 38.75% | 40.42% | 9.68 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 558 | 223 | 335 | 39.96% | 40.83% | 38.96% | 10.04 pp | -112 | 48 | -2.33 |
| BTC Daily | lstm | LSTM | 732 | 315 | 417 | 43.03% | 37.08% | 41.25% | 6.97 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 732 | 312 | 420 | 42.62% | 40.83% | 43.33% | 7.38 pp | -108 | 43 | -2.51 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 909 | 389 | 520 | 42.79% | 39.58% | 41.88% | 7.21 pp | -131 | 48 | -2.73 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| BTC Hourly | xgb | XGBoost | 909 | 381 | 528 | 41.91% | 40.83% | 40.83% | 8.09 pp | -147 | 48 | -3.06 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 742 | 293 | 449 | 39.49% | 35.83% | 38.12% | 10.51 pp | -156 | 43 | -3.63 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 909 | 436 | 473 | 47.96% | 51.25% | 48.33% | 2.04 pp | -37 | 48 | -0.77 |
| BTC Hourly | transformer | Transformer | 909 | 429 | 480 | 47.19% | 47.50% | 46.67% | 2.81 pp | -51 | 48 | -1.06 |
| BTC Hourly | nn | NN | 909 | 404 | 505 | 44.44% | 43.75% | 42.08% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | rf | RandomForest | 909 | 404 | 505 | 44.44% | 44.17% | 43.96% | 5.56 pp | -101 | 48 | -2.10 |
| BTC Hourly | lstm | LSTM | 909 | 389 | 520 | 42.79% | 39.58% | 41.88% | 7.21 pp | -131 | 48 | -2.73 |
| BTC Hourly | xgb | XGBoost | 909 | 381 | 528 | 41.91% | 40.83% | 40.83% | 8.09 pp | -147 | 48 | -3.06 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 732 | 354 | 378 | 48.36% | 47.08% | 48.12% | 1.64 pp | -24 | 43 | -0.56 |
| BTC Daily | transformer | Transformer | 732 | 349 | 383 | 47.68% | 46.67% | 49.58% | 2.32 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 732 | 339 | 393 | 46.31% | 45.00% | 47.08% | 3.69 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 732 | 315 | 417 | 43.03% | 37.08% | 41.25% | 6.97 pp | -102 | 43 | -2.37 |
| BTC Daily | rf | RandomForest | 732 | 312 | 420 | 42.62% | 40.83% | 43.33% | 7.38 pp | -108 | 43 | -2.51 |
| BTC Daily | xgb | XGBoost | 742 | 293 | 449 | 39.49% | 35.83% | 38.12% | 10.51 pp | -156 | 43 | -3.63 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 504 | 242 | 262 | 48.02% | 44.58% | 47.92% | 1.98 pp | -20 | 48 | -0.42 |
| BTC Market Hours | transformer | Transformer | 504 | 240 | 264 | 47.62% | 46.25% | 48.12% | 2.38 pp | -24 | 48 | -0.50 |
| BTC Market Hours | nn | NN | 504 | 239 | 265 | 47.42% | 50.83% | 48.33% | 2.58 pp | -26 | 48 | -0.54 |
| BTC Market Hours | lstm | LSTM | 504 | 217 | 287 | 43.06% | 41.25% | 43.12% | 6.94 pp | -70 | 48 | -1.46 |
| BTC Market Hours | rf | RandomForest | 504 | 216 | 288 | 42.86% | 44.17% | 43.12% | 7.14 pp | -72 | 48 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 504 | 208 | 296 | 41.27% | 42.50% | 41.88% | 8.73 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 558 | 263 | 295 | 47.13% | 50.00% | 47.92% | 2.87 pp | -32 | 48 | -0.67 |
| BTC Market Hours Daily | nn | NN | 558 | 259 | 299 | 46.42% | 45.83% | 47.71% | 3.58 pp | -40 | 48 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 558 | 258 | 300 | 46.24% | 49.58% | 46.67% | 3.76 pp | -42 | 48 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 558 | 231 | 327 | 41.40% | 42.08% | 40.42% | 8.60 pp | -96 | 48 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 558 | 225 | 333 | 40.32% | 38.75% | 40.42% | 9.68 pp | -108 | 48 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 558 | 223 | 335 | 39.96% | 40.83% | 38.96% | 10.04 pp | -112 | 48 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 152 | 75 | 77 | 49.34% | 49.34% | 49.34% | 0.66 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | xgb | XGBoost | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | nn | NN | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 11 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 152 | 64 | 88 | 42.11% | 42.11% | 42.11% | 7.89 pp | -24 | 11 | -2.18 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 152 | 75 | 77 | 49.34% | 49.34% | 49.34% | 0.66 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 152 | 72 | 80 | 47.37% | 47.37% | 47.37% | 2.63 pp | -8 | 11 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 152 | 69 | 83 | 45.39% | 45.39% | 45.39% | 4.61 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 152 | 66 | 86 | 43.42% | 43.42% | 43.42% | 6.58 pp | -20 | 11 | -1.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 152 | 64 | 88 | 42.11% | 42.11% | 42.11% | 7.89 pp | -24 | 11 | -2.18 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours | rf | RandomForest | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 37 | 14 | 23 | 37.84% | 37.84% | 37.84% | 12.16 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
