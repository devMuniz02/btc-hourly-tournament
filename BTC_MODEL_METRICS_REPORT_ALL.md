# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T00:29:29.388948+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1187 | 899 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1062 | 697 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 703 | 459 | 243 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 23:00:00+00:00 | 705 | 513 | 190 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 459 | 223 | 236 | 48.58% | 44.58% | 48.58% | 1.42 pp | -13 | 45 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 687 | 335 | 352 | 48.76% | 45.83% | 49.38% | 1.24 pp | -17 | 41 | -0.41 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 459 | 216 | 243 | 47.06% | 48.75% | 47.06% | 2.94 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 687 | 330 | 357 | 48.03% | 46.25% | 49.38% | 1.97 pp | -27 | 41 | -0.66 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 459 | 212 | 247 | 46.19% | 40.00% | 46.19% | 3.81 pp | -35 | 45 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 513 | 235 | 278 | 45.81% | 46.25% | 46.25% | 4.19 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | nn | NN | 513 | 235 | 278 | 45.81% | 43.33% | 46.67% | 4.19 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 513 | 233 | 280 | 45.42% | 46.67% | 45.83% | 4.58 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 687 | 321 | 366 | 46.72% | 43.33% | 49.17% | 3.28 pp | -45 | 41 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 865 | 407 | 458 | 47.05% | 45.42% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Hourly | transformer | Transformer | 865 | 407 | 458 | 47.05% | 47.50% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Market Hours | rf | RandomForest | 459 | 200 | 259 | 43.57% | 43.75% | 43.57% | 6.43 pp | -59 | 45 | -1.31 |
| BTC Market Hours | lstm | LSTM | 459 | 195 | 264 | 42.48% | 40.00% | 42.48% | 7.52 pp | -69 | 45 | -1.53 |
| BTC Market Hours Daily | rf | RandomForest | 513 | 214 | 299 | 41.72% | 42.08% | 41.88% | 8.28 pp | -85 | 45 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 459 | 187 | 272 | 40.74% | 39.58% | 40.74% | 9.26 pp | -85 | 45 | -1.89 |
| BTC Hourly | nn | NN | 865 | 389 | 476 | 44.97% | 45.42% | 43.96% | 5.03 pp | -87 | 46 | -1.89 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| BTC Hourly | rf | RandomForest | 865 | 384 | 481 | 44.39% | 43.33% | 43.96% | 5.61 pp | -97 | 46 | -2.11 |
| BTC Daily | lstm | LSTM | 687 | 300 | 387 | 43.67% | 39.17% | 42.71% | 6.33 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 513 | 206 | 307 | 40.16% | 38.33% | 40.83% | 9.84 pp | -101 | 45 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 513 | 204 | 309 | 39.77% | 37.08% | 39.38% | 10.23 pp | -105 | 45 | -2.33 |
| BTC Daily | rf | RandomForest | 687 | 295 | 392 | 42.94% | 40.42% | 43.33% | 7.06 pp | -97 | 41 | -2.37 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 865 | 367 | 498 | 42.43% | 37.92% | 41.88% | 7.57 pp | -131 | 46 | -2.85 |
| BTC Hourly | xgb | XGBoost | 865 | 364 | 501 | 42.08% | 40.42% | 42.92% | 7.92 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 697 | 276 | 421 | 39.60% | 35.42% | 39.17% | 10.40 pp | -145 | 41 | -3.54 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 865 | 407 | 458 | 47.05% | 45.42% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Hourly | transformer | Transformer | 865 | 407 | 458 | 47.05% | 47.50% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 865 | 389 | 476 | 44.97% | 45.42% | 43.96% | 5.03 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 865 | 384 | 481 | 44.39% | 43.33% | 43.96% | 5.61 pp | -97 | 46 | -2.11 |
| BTC Hourly | lstm | LSTM | 865 | 367 | 498 | 42.43% | 37.92% | 41.88% | 7.57 pp | -131 | 46 | -2.85 |
| BTC Hourly | xgb | XGBoost | 865 | 364 | 501 | 42.08% | 40.42% | 42.92% | 7.92 pp | -137 | 46 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 687 | 335 | 352 | 48.76% | 45.83% | 49.38% | 1.24 pp | -17 | 41 | -0.41 |
| BTC Daily | transformer | Transformer | 687 | 330 | 357 | 48.03% | 46.25% | 49.38% | 1.97 pp | -27 | 41 | -0.66 |
| BTC Daily | nn | NN | 687 | 321 | 366 | 46.72% | 43.33% | 49.17% | 3.28 pp | -45 | 41 | -1.10 |
| BTC Daily | lstm | LSTM | 687 | 300 | 387 | 43.67% | 39.17% | 42.71% | 6.33 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 687 | 295 | 392 | 42.94% | 40.42% | 43.33% | 7.06 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 697 | 276 | 421 | 39.60% | 35.42% | 39.17% | 10.40 pp | -145 | 41 | -3.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 459 | 223 | 236 | 48.58% | 44.58% | 48.58% | 1.42 pp | -13 | 45 | -0.29 |
| BTC Market Hours | nn | NN | 459 | 216 | 243 | 47.06% | 48.75% | 47.06% | 2.94 pp | -27 | 45 | -0.60 |
| BTC Market Hours | transformer | Transformer | 459 | 212 | 247 | 46.19% | 40.00% | 46.19% | 3.81 pp | -35 | 45 | -0.78 |
| BTC Market Hours | rf | RandomForest | 459 | 200 | 259 | 43.57% | 43.75% | 43.57% | 6.43 pp | -59 | 45 | -1.31 |
| BTC Market Hours | lstm | LSTM | 459 | 195 | 264 | 42.48% | 40.00% | 42.48% | 7.52 pp | -69 | 45 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 459 | 187 | 272 | 40.74% | 39.58% | 40.74% | 9.26 pp | -85 | 45 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 513 | 235 | 278 | 45.81% | 46.25% | 46.25% | 4.19 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | nn | NN | 513 | 235 | 278 | 45.81% | 43.33% | 46.67% | 4.19 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 513 | 233 | 280 | 45.42% | 46.67% | 45.83% | 4.58 pp | -47 | 45 | -1.04 |
| BTC Market Hours Daily | rf | RandomForest | 513 | 214 | 299 | 41.72% | 42.08% | 41.88% | 8.28 pp | -85 | 45 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 513 | 206 | 307 | 40.16% | 38.33% | 40.83% | 9.84 pp | -101 | 45 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 513 | 204 | 309 | 39.77% | 37.08% | 39.38% | 10.23 pp | -105 | 45 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
