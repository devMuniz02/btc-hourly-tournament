# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T10:21:08.496789+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1143 | 855 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1019 | 654 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 622 | 416 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 624 | 470 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 23:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 23:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 23:00:00+00:00 | 72 | 1 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 23:00:00+00:00 | 72 | 1 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 7 | 1.14 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 416 | 206 | 210 | 49.52% | 47.92% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 644 | 313 | 331 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 644 | 313 | 331 | 48.60% | 45.83% | 49.38% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Market Hours | nn | NN | 416 | 197 | 219 | 47.36% | 50.83% | 47.36% | 2.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 470 | 218 | 252 | 46.38% | 46.67% | 46.38% | 3.62 pp | -34 | 41 | -0.83 |
| BTC Market Hours | transformer | Transformer | 416 | 191 | 225 | 45.91% | 41.67% | 45.91% | 4.09 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 470 | 215 | 255 | 45.74% | 45.42% | 45.74% | 4.26 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 821 | 389 | 432 | 47.38% | 47.08% | 46.46% | 2.62 pp | -43 | 44 | -0.98 |
| BTC Daily | nn | NN | 644 | 302 | 342 | 46.89% | 42.50% | 49.17% | 3.11 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 470 | 214 | 256 | 45.53% | 44.58% | 45.53% | 4.47 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 821 | 386 | 435 | 47.02% | 42.92% | 46.88% | 2.98 pp | -49 | 44 | -1.11 |
| Consolidated Hourly | xgb | XGBoost | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 416 | 183 | 233 | 43.99% | 44.17% | 43.99% | 6.01 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 416 | 180 | 236 | 43.27% | 42.50% | 43.27% | 6.73 pp | -56 | 41 | -1.37 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 7 | -1.71 |
| BTC Hourly | nn | NN | 821 | 371 | 450 | 45.19% | 42.50% | 45.00% | 4.81 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 644 | 285 | 359 | 44.25% | 41.67% | 43.75% | 5.75 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 470 | 195 | 275 | 41.49% | 42.50% | 41.49% | 8.51 pp | -80 | 41 | -1.95 |
| BTC Hourly | rf | RandomForest | 821 | 367 | 454 | 44.70% | 44.58% | 44.38% | 5.30 pp | -87 | 44 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 37.50% | 39.90% | 10.10 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 470 | 192 | 278 | 40.85% | 39.58% | 40.85% | 9.15 pp | -86 | 41 | -2.10 |
| Consolidated Hourly | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 7 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 7 | -2.29 |
| BTC Daily | rf | RandomForest | 644 | 275 | 369 | 42.70% | 41.25% | 43.54% | 7.30 pp | -94 | 40 | -2.35 |
| BTC Hourly | lstm | LSTM | 821 | 357 | 464 | 43.48% | 41.67% | 43.96% | 6.52 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 470 | 182 | 288 | 38.72% | 35.42% | 38.72% | 11.28 pp | -106 | 41 | -2.59 |
| BTC Hourly | xgb | XGBoost | 821 | 347 | 474 | 42.27% | 39.58% | 42.50% | 7.73 pp | -127 | 44 | -2.89 |
| BTC Daily | xgb | XGBoost | 654 | 257 | 397 | 39.30% | 31.25% | 39.38% | 10.70 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 821 | 389 | 432 | 47.38% | 47.08% | 46.46% | 2.62 pp | -43 | 44 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 821 | 386 | 435 | 47.02% | 42.92% | 46.88% | 2.98 pp | -49 | 44 | -1.11 |
| BTC Hourly | nn | NN | 821 | 371 | 450 | 45.19% | 42.50% | 45.00% | 4.81 pp | -79 | 44 | -1.80 |
| BTC Hourly | rf | RandomForest | 821 | 367 | 454 | 44.70% | 44.58% | 44.38% | 5.30 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 821 | 357 | 464 | 43.48% | 41.67% | 43.96% | 6.52 pp | -107 | 44 | -2.43 |
| BTC Hourly | xgb | XGBoost | 821 | 347 | 474 | 42.27% | 39.58% | 42.50% | 7.73 pp | -127 | 44 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 644 | 313 | 331 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 644 | 313 | 331 | 48.60% | 45.83% | 49.38% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | nn | NN | 644 | 302 | 342 | 46.89% | 42.50% | 49.17% | 3.11 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 644 | 285 | 359 | 44.25% | 41.67% | 43.75% | 5.75 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 644 | 275 | 369 | 42.70% | 41.25% | 43.54% | 7.30 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 654 | 257 | 397 | 39.30% | 31.25% | 39.38% | 10.70 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 416 | 206 | 210 | 49.52% | 47.92% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Market Hours | nn | NN | 416 | 197 | 219 | 47.36% | 50.83% | 47.36% | 2.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 416 | 191 | 225 | 45.91% | 41.67% | 45.91% | 4.09 pp | -34 | 41 | -0.83 |
| BTC Market Hours | lstm | LSTM | 416 | 183 | 233 | 43.99% | 44.17% | 43.99% | 6.01 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 416 | 180 | 236 | 43.27% | 42.50% | 43.27% | 6.73 pp | -56 | 41 | -1.37 |
| BTC Market Hours | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 37.50% | 39.90% | 10.10 pp | -84 | 41 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 470 | 218 | 252 | 46.38% | 46.67% | 46.38% | 3.62 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 470 | 215 | 255 | 45.74% | 45.42% | 45.74% | 4.26 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | nn | NN | 470 | 214 | 256 | 45.53% | 44.58% | 45.53% | 4.47 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 470 | 195 | 275 | 41.49% | 42.50% | 41.49% | 8.51 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 470 | 192 | 278 | 40.85% | 39.58% | 40.85% | 9.15 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 470 | 182 | 288 | 38.72% | 35.42% | 38.72% | 11.28 pp | -106 | 41 | -2.59 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 7 | -2.29 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 37 | 35 | 51.39% | 51.39% | 51.39% | 1.39 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 7 | -2.29 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
