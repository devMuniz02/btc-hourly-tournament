# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T01:11:18.946898+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1063 | 698 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 705 | 460 | 244 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 707 | 514 | 191 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 110 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 110 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 110 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T12:00:00+00:00 | 111 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 10 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 460 | 224 | 236 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 688 | 336 | 352 | 48.84% | 45.83% | 49.38% | 1.16 pp | -16 | 41 | -0.39 |
| Consolidated Hourly | xgb | XGBoost | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 460 | 217 | 243 | 47.17% | 48.75% | 47.17% | 2.83 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 688 | 331 | 357 | 48.11% | 46.25% | 49.58% | 1.89 pp | -26 | 41 | -0.63 |
| BTC Market Hours | transformer | Transformer | 460 | 213 | 247 | 46.30% | 40.42% | 46.30% | 3.70 pp | -34 | 45 | -0.76 |
| Consolidated Hourly | lstm | LSTM | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 10 | -0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 514 | 236 | 278 | 45.91% | 46.25% | 46.25% | 4.09 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | nn | NN | 514 | 235 | 279 | 45.72% | 42.92% | 46.46% | 4.28 pp | -44 | 45 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 514 | 234 | 280 | 45.53% | 46.67% | 46.04% | 4.47 pp | -46 | 45 | -1.02 |
| BTC Daily | nn | NN | 688 | 322 | 366 | 46.80% | 43.33% | 49.38% | 3.20 pp | -44 | 41 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 865 | 407 | 458 | 47.05% | 45.42% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Hourly | transformer | Transformer | 865 | 407 | 458 | 47.05% | 47.50% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| Consolidated Hourly | nn | NN | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |
| BTC Market Hours | rf | RandomForest | 460 | 200 | 260 | 43.48% | 43.75% | 43.48% | 6.52 pp | -60 | 45 | -1.33 |
| BTC Market Hours | lstm | LSTM | 460 | 195 | 265 | 42.39% | 40.00% | 42.39% | 7.61 pp | -70 | 45 | -1.56 |
| BTC Market Hours Daily | rf | RandomForest | 514 | 215 | 299 | 41.83% | 42.08% | 42.08% | 8.17 pp | -84 | 45 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 460 | 188 | 272 | 40.87% | 40.00% | 40.87% | 9.13 pp | -84 | 45 | -1.87 |
| BTC Hourly | nn | NN | 865 | 389 | 476 | 44.97% | 45.42% | 43.96% | 5.03 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 865 | 384 | 481 | 44.39% | 43.33% | 43.96% | 5.61 pp | -97 | 46 | -2.11 |
| BTC Daily | lstm | LSTM | 688 | 300 | 388 | 43.60% | 38.75% | 42.71% | 6.40 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 514 | 206 | 308 | 40.08% | 37.92% | 40.83% | 9.92 pp | -102 | 45 | -2.27 |
| BTC Daily | rf | RandomForest | 688 | 296 | 392 | 43.02% | 40.42% | 43.54% | 6.98 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 514 | 204 | 310 | 39.69% | 36.67% | 39.38% | 10.31 pp | -106 | 45 | -2.36 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Hourly | lstm | LSTM | 865 | 367 | 498 | 42.43% | 37.92% | 41.88% | 7.57 pp | -131 | 46 | -2.85 |
| BTC Hourly | xgb | XGBoost | 865 | 364 | 501 | 42.08% | 40.42% | 42.92% | 7.92 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | xgb | XGBoost | 698 | 277 | 421 | 39.68% | 35.83% | 39.38% | 10.32 pp | -144 | 41 | -3.51 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 688 | 336 | 352 | 48.84% | 45.83% | 49.38% | 1.16 pp | -16 | 41 | -0.39 |
| BTC Daily | transformer | Transformer | 688 | 331 | 357 | 48.11% | 46.25% | 49.58% | 1.89 pp | -26 | 41 | -0.63 |
| BTC Daily | nn | NN | 688 | 322 | 366 | 46.80% | 43.33% | 49.38% | 3.20 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 688 | 300 | 388 | 43.60% | 38.75% | 42.71% | 6.40 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 688 | 296 | 392 | 43.02% | 40.42% | 43.54% | 6.98 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 698 | 277 | 421 | 39.68% | 35.83% | 39.38% | 10.32 pp | -144 | 41 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 460 | 224 | 236 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| BTC Market Hours | nn | NN | 460 | 217 | 243 | 47.17% | 48.75% | 47.17% | 2.83 pp | -26 | 45 | -0.58 |
| BTC Market Hours | transformer | Transformer | 460 | 213 | 247 | 46.30% | 40.42% | 46.30% | 3.70 pp | -34 | 45 | -0.76 |
| BTC Market Hours | rf | RandomForest | 460 | 200 | 260 | 43.48% | 43.75% | 43.48% | 6.52 pp | -60 | 45 | -1.33 |
| BTC Market Hours | lstm | LSTM | 460 | 195 | 265 | 42.39% | 40.00% | 42.39% | 7.61 pp | -70 | 45 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 460 | 188 | 272 | 40.87% | 40.00% | 40.87% | 9.13 pp | -84 | 45 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 514 | 236 | 278 | 45.91% | 46.25% | 46.25% | 4.09 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | nn | NN | 514 | 235 | 279 | 45.72% | 42.92% | 46.46% | 4.28 pp | -44 | 45 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 514 | 234 | 280 | 45.53% | 46.67% | 46.04% | 4.47 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 514 | 215 | 299 | 41.83% | 42.08% | 42.08% | 8.17 pp | -84 | 45 | -1.87 |
| BTC Market Hours Daily | lstm | LSTM | 514 | 206 | 308 | 40.08% | 37.92% | 40.83% | 9.92 pp | -102 | 45 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 514 | 204 | 310 | 39.69% | 36.67% | 39.38% | 10.31 pp | -106 | 45 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 10 | -0.20 |
| Consolidated Hourly | xgb | XGBoost | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 10 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 10 | -0.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 110 | 51 | 59 | 46.36% | 46.36% | 46.36% | 3.64 pp | -8 | 10 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
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
