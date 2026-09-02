# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T03:28:27.027426+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1189 | 901 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1064 | 699 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 706 | 461 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 708 | 515 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 111 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 111 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 16 | 95 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 13:00:00+00:00 | 111 | 16 | 95 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 461 | 225 | 236 | 48.81% | 45.00% | 48.81% | 1.19 pp | -11 | 45 | -0.24 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 689 | 337 | 352 | 48.91% | 45.83% | 49.58% | 1.09 pp | -15 | 42 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| BTC Market Hours | nn | NN | 461 | 217 | 244 | 47.07% | 48.33% | 47.07% | 2.93 pp | -27 | 45 | -0.60 |
| BTC Daily | transformer | Transformer | 689 | 331 | 358 | 48.04% | 46.25% | 49.38% | 1.96 pp | -27 | 42 | -0.64 |
| Consolidated Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 461 | 213 | 248 | 46.20% | 40.42% | 46.20% | 3.80 pp | -35 | 45 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 515 | 237 | 278 | 46.02% | 46.67% | 46.46% | 3.98 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 515 | 236 | 279 | 45.83% | 43.33% | 46.67% | 4.17 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 515 | 235 | 280 | 45.63% | 47.08% | 46.04% | 4.37 pp | -45 | 45 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 867 | 409 | 458 | 47.17% | 46.25% | 47.50% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 689 | 321 | 368 | 46.59% | 42.92% | 48.96% | 3.41 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 867 | 407 | 460 | 46.94% | 47.50% | 46.88% | 3.06 pp | -53 | 46 | -1.15 |
| Consolidated Hourly | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 461 | 200 | 261 | 43.38% | 43.75% | 43.38% | 6.62 pp | -61 | 45 | -1.36 |
| BTC Market Hours | lstm | LSTM | 461 | 196 | 265 | 42.52% | 40.00% | 42.52% | 7.48 pp | -69 | 45 | -1.53 |
| BTC Market Hours Daily | rf | RandomForest | 515 | 215 | 300 | 41.75% | 42.08% | 41.88% | 8.25 pp | -85 | 45 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 45 | -1.89 |
| BTC Hourly | nn | NN | 867 | 390 | 477 | 44.98% | 45.83% | 44.17% | 5.02 pp | -87 | 46 | -1.89 |
| Consolidated Hourly | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 867 | 386 | 481 | 44.52% | 44.17% | 44.38% | 5.48 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 689 | 301 | 388 | 43.69% | 39.17% | 42.71% | 6.31 pp | -87 | 42 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 515 | 206 | 309 | 40.00% | 37.92% | 40.62% | 10.00 pp | -103 | 45 | -2.29 |
| BTC Daily | rf | RandomForest | 689 | 296 | 393 | 42.96% | 40.00% | 43.33% | 7.04 pp | -97 | 42 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 515 | 205 | 310 | 39.81% | 37.08% | 39.38% | 10.19 pp | -105 | 45 | -2.33 |
| BTC Hourly | lstm | LSTM | 867 | 369 | 498 | 42.56% | 38.33% | 42.29% | 7.44 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 867 | 365 | 502 | 42.10% | 40.42% | 43.12% | 7.90 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 699 | 277 | 422 | 39.63% | 35.83% | 39.38% | 10.37 pp | -145 | 42 | -3.45 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 867 | 409 | 458 | 47.17% | 46.25% | 47.50% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 867 | 407 | 460 | 46.94% | 47.50% | 46.88% | 3.06 pp | -53 | 46 | -1.15 |
| BTC Hourly | nn | NN | 867 | 390 | 477 | 44.98% | 45.83% | 44.17% | 5.02 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 867 | 386 | 481 | 44.52% | 44.17% | 44.38% | 5.48 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 867 | 369 | 498 | 42.56% | 38.33% | 42.29% | 7.44 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 867 | 365 | 502 | 42.10% | 40.42% | 43.12% | 7.90 pp | -137 | 46 | -2.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 689 | 337 | 352 | 48.91% | 45.83% | 49.58% | 1.09 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 689 | 331 | 358 | 48.04% | 46.25% | 49.38% | 1.96 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 689 | 321 | 368 | 46.59% | 42.92% | 48.96% | 3.41 pp | -47 | 42 | -1.12 |
| BTC Daily | lstm | LSTM | 689 | 301 | 388 | 43.69% | 39.17% | 42.71% | 6.31 pp | -87 | 42 | -2.07 |
| BTC Daily | rf | RandomForest | 689 | 296 | 393 | 42.96% | 40.00% | 43.33% | 7.04 pp | -97 | 42 | -2.31 |
| BTC Daily | xgb | XGBoost | 699 | 277 | 422 | 39.63% | 35.83% | 39.38% | 10.37 pp | -145 | 42 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 461 | 225 | 236 | 48.81% | 45.00% | 48.81% | 1.19 pp | -11 | 45 | -0.24 |
| BTC Market Hours | nn | NN | 461 | 217 | 244 | 47.07% | 48.33% | 47.07% | 2.93 pp | -27 | 45 | -0.60 |
| BTC Market Hours | transformer | Transformer | 461 | 213 | 248 | 46.20% | 40.42% | 46.20% | 3.80 pp | -35 | 45 | -0.78 |
| BTC Market Hours | rf | RandomForest | 461 | 200 | 261 | 43.38% | 43.75% | 43.38% | 6.62 pp | -61 | 45 | -1.36 |
| BTC Market Hours | lstm | LSTM | 461 | 196 | 265 | 42.52% | 40.00% | 42.52% | 7.48 pp | -69 | 45 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 461 | 188 | 273 | 40.78% | 40.00% | 40.78% | 9.22 pp | -85 | 45 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 515 | 237 | 278 | 46.02% | 46.67% | 46.46% | 3.98 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 515 | 236 | 279 | 45.83% | 43.33% | 46.67% | 4.17 pp | -43 | 45 | -0.96 |
| BTC Market Hours Daily | transformer | Transformer | 515 | 235 | 280 | 45.63% | 47.08% | 46.04% | 4.37 pp | -45 | 45 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 515 | 215 | 300 | 41.75% | 42.08% | 41.88% | 8.25 pp | -85 | 45 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 515 | 206 | 309 | 40.00% | 37.92% | 40.62% | 10.00 pp | -103 | 45 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 515 | 205 | 310 | 39.81% | 37.08% | 39.38% | 10.19 pp | -105 | 45 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 111 | 58 | 53 | 52.25% | 52.25% | 52.25% | 2.25 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
