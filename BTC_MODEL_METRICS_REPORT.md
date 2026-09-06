# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T00:48:54.133460+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1252 | 964 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1128 | 763 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 23:00:00+00:00 | 821 | 525 | 295 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 23:00:00+00:00 | 823 | 579 | 242 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 525 | 255 | 270 | 48.57% | 45.42% | 48.54% | 1.43 pp | -15 | 50 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 753 | 367 | 386 | 48.74% | 48.33% | 48.96% | 1.26 pp | -19 | 44 | -0.43 |
| BTC Market Hours | transformer | Transformer | 525 | 251 | 274 | 47.81% | 47.50% | 48.33% | 2.19 pp | -23 | 50 | -0.46 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 579 | 275 | 304 | 47.50% | 50.42% | 48.75% | 2.50 pp | -29 | 50 | -0.58 |
| BTC Market Hours | nn | NN | 525 | 247 | 278 | 47.05% | 50.42% | 48.54% | 2.95 pp | -31 | 50 | -0.62 |
| BTC Market Hours Daily | nn | NN | 579 | 270 | 309 | 46.63% | 46.25% | 47.92% | 3.37 pp | -39 | 50 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 930 | 443 | 487 | 47.63% | 49.17% | 46.67% | 2.37 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 579 | 267 | 312 | 46.11% | 50.42% | 46.67% | 3.89 pp | -45 | 50 | -0.90 |
| BTC Daily | transformer | Transformer | 753 | 356 | 397 | 47.28% | 44.17% | 48.75% | 2.72 pp | -41 | 44 | -0.93 |
| BTC Daily | nn | NN | 753 | 351 | 402 | 46.61% | 45.42% | 46.88% | 3.39 pp | -51 | 44 | -1.16 |
| BTC Hourly | transformer | Transformer | 930 | 436 | 494 | 46.88% | 45.83% | 45.21% | 3.12 pp | -58 | 49 | -1.18 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| BTC Market Hours | lstm | LSTM | 525 | 228 | 297 | 43.43% | 42.50% | 44.17% | 6.57 pp | -69 | 50 | -1.38 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | rf | RandomForest | 525 | 225 | 300 | 42.86% | 44.17% | 43.54% | 7.14 pp | -75 | 50 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 525 | 217 | 308 | 41.33% | 42.92% | 41.88% | 8.67 pp | -91 | 50 | -1.82 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 579 | 240 | 339 | 41.45% | 43.33% | 41.04% | 8.55 pp | -99 | 50 | -1.98 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 930 | 414 | 516 | 44.52% | 43.75% | 44.38% | 5.48 pp | -102 | 49 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 579 | 236 | 343 | 40.76% | 41.25% | 40.62% | 9.24 pp | -107 | 50 | -2.14 |
| BTC Hourly | nn | NN | 930 | 411 | 519 | 44.19% | 41.67% | 41.88% | 5.81 pp | -108 | 49 | -2.20 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 579 | 231 | 348 | 39.90% | 41.67% | 39.17% | 10.10 pp | -117 | 50 | -2.34 |
| BTC Daily | lstm | LSTM | 753 | 320 | 433 | 42.50% | 35.83% | 40.83% | 7.50 pp | -113 | 44 | -2.57 |
| BTC Daily | rf | RandomForest | 753 | 316 | 437 | 41.97% | 38.33% | 42.08% | 8.03 pp | -121 | 44 | -2.75 |
| BTC Hourly | lstm | LSTM | 930 | 396 | 534 | 42.58% | 37.50% | 41.25% | 7.42 pp | -138 | 49 | -2.82 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 930 | 389 | 541 | 41.83% | 39.58% | 40.83% | 8.17 pp | -152 | 49 | -3.10 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 763 | 299 | 464 | 39.19% | 35.42% | 37.08% | 10.81 pp | -165 | 44 | -3.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 930 | 443 | 487 | 47.63% | 49.17% | 46.67% | 2.37 pp | -44 | 49 | -0.90 |
| BTC Hourly | transformer | Transformer | 930 | 436 | 494 | 46.88% | 45.83% | 45.21% | 3.12 pp | -58 | 49 | -1.18 |
| BTC Hourly | rf | RandomForest | 930 | 414 | 516 | 44.52% | 43.75% | 44.38% | 5.48 pp | -102 | 49 | -2.08 |
| BTC Hourly | nn | NN | 930 | 411 | 519 | 44.19% | 41.67% | 41.88% | 5.81 pp | -108 | 49 | -2.20 |
| BTC Hourly | lstm | LSTM | 930 | 396 | 534 | 42.58% | 37.50% | 41.25% | 7.42 pp | -138 | 49 | -2.82 |
| BTC Hourly | xgb | XGBoost | 930 | 389 | 541 | 41.83% | 39.58% | 40.83% | 8.17 pp | -152 | 49 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 753 | 367 | 386 | 48.74% | 48.33% | 48.96% | 1.26 pp | -19 | 44 | -0.43 |
| BTC Daily | transformer | Transformer | 753 | 356 | 397 | 47.28% | 44.17% | 48.75% | 2.72 pp | -41 | 44 | -0.93 |
| BTC Daily | nn | NN | 753 | 351 | 402 | 46.61% | 45.42% | 46.88% | 3.39 pp | -51 | 44 | -1.16 |
| BTC Daily | lstm | LSTM | 753 | 320 | 433 | 42.50% | 35.83% | 40.83% | 7.50 pp | -113 | 44 | -2.57 |
| BTC Daily | rf | RandomForest | 753 | 316 | 437 | 41.97% | 38.33% | 42.08% | 8.03 pp | -121 | 44 | -2.75 |
| BTC Daily | xgb | XGBoost | 763 | 299 | 464 | 39.19% | 35.42% | 37.08% | 10.81 pp | -165 | 44 | -3.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 525 | 255 | 270 | 48.57% | 45.42% | 48.54% | 1.43 pp | -15 | 50 | -0.30 |
| BTC Market Hours | transformer | Transformer | 525 | 251 | 274 | 47.81% | 47.50% | 48.33% | 2.19 pp | -23 | 50 | -0.46 |
| BTC Market Hours | nn | NN | 525 | 247 | 278 | 47.05% | 50.42% | 48.54% | 2.95 pp | -31 | 50 | -0.62 |
| BTC Market Hours | lstm | LSTM | 525 | 228 | 297 | 43.43% | 42.50% | 44.17% | 6.57 pp | -69 | 50 | -1.38 |
| BTC Market Hours | rf | RandomForest | 525 | 225 | 300 | 42.86% | 44.17% | 43.54% | 7.14 pp | -75 | 50 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 525 | 217 | 308 | 41.33% | 42.92% | 41.88% | 8.67 pp | -91 | 50 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 579 | 275 | 304 | 47.50% | 50.42% | 48.75% | 2.50 pp | -29 | 50 | -0.58 |
| BTC Market Hours Daily | nn | NN | 579 | 270 | 309 | 46.63% | 46.25% | 47.92% | 3.37 pp | -39 | 50 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 579 | 267 | 312 | 46.11% | 50.42% | 46.67% | 3.89 pp | -45 | 50 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 579 | 240 | 339 | 41.45% | 43.33% | 41.04% | 8.55 pp | -99 | 50 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 579 | 236 | 343 | 40.76% | 41.25% | 40.62% | 9.24 pp | -107 | 50 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 579 | 231 | 348 | 39.90% | 41.67% | 39.17% | 10.10 pp | -117 | 50 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
