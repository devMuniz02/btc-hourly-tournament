# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T01:50:12.621694+00:00
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 460 | 224 | 236 | 48.70% | 44.58% | 48.70% | 1.30 pp | -12 | 45 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 111 | 54 | 57 | 48.65% | 48.65% | 48.65% | 1.35 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 688 | 336 | 352 | 48.84% | 45.83% | 49.38% | 1.16 pp | -16 | 41 | -0.39 |
| Consolidated Hourly | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 10 | -0.50 |
| BTC Market Hours | nn | NN | 460 | 217 | 243 | 47.17% | 48.75% | 47.17% | 2.83 pp | -26 | 45 | -0.58 |
| BTC Daily | transformer | Transformer | 688 | 331 | 357 | 48.11% | 46.25% | 49.58% | 1.89 pp | -26 | 41 | -0.63 |
| Consolidated Hourly | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 111 | 52 | 59 | 46.85% | 46.85% | 46.85% | 3.15 pp | -7 | 10 | -0.70 |
| BTC Market Hours | transformer | Transformer | 460 | 213 | 247 | 46.30% | 40.42% | 46.30% | 3.70 pp | -34 | 45 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 514 | 236 | 278 | 45.91% | 46.25% | 46.25% | 4.09 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | nn | NN | 514 | 235 | 279 | 45.72% | 42.92% | 46.46% | 4.28 pp | -44 | 45 | -0.98 |
| Consolidated Market Hours | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 514 | 234 | 280 | 45.53% | 46.67% | 46.04% | 4.47 pp | -46 | 45 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 865 | 408 | 457 | 47.17% | 45.83% | 47.29% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Daily | nn | NN | 688 | 322 | 366 | 46.80% | 43.33% | 49.38% | 3.20 pp | -44 | 41 | -1.07 |
| BTC Hourly | transformer | Transformer | 865 | 407 | 458 | 47.05% | 47.50% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| Consolidated Hourly | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 10 | -1.30 |
| BTC Market Hours | rf | RandomForest | 460 | 200 | 260 | 43.48% | 43.75% | 43.48% | 6.52 pp | -60 | 45 | -1.33 |
| BTC Market Hours | lstm | LSTM | 460 | 195 | 265 | 42.39% | 40.00% | 42.39% | 7.61 pp | -70 | 45 | -1.56 |
| BTC Market Hours Daily | rf | RandomForest | 514 | 215 | 299 | 41.83% | 42.08% | 42.08% | 8.17 pp | -84 | 45 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 460 | 188 | 272 | 40.87% | 40.00% | 40.87% | 9.13 pp | -84 | 45 | -1.87 |
| BTC Hourly | nn | NN | 865 | 389 | 476 | 44.97% | 45.42% | 43.96% | 5.03 pp | -87 | 46 | -1.89 |
| Consolidated Hourly | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 10 | -1.90 |
| Consolidated Market Hours | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 16 | 6 | 10 | 37.50% | 37.50% | 37.50% | 12.50 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 865 | 385 | 480 | 44.51% | 43.75% | 44.17% | 5.49 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 688 | 300 | 388 | 43.60% | 38.75% | 42.71% | 6.40 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 514 | 206 | 308 | 40.08% | 37.92% | 40.83% | 9.92 pp | -102 | 45 | -2.27 |
| BTC Daily | rf | RandomForest | 688 | 296 | 392 | 43.02% | 40.42% | 43.54% | 6.98 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 514 | 204 | 310 | 39.69% | 36.67% | 39.38% | 10.31 pp | -106 | 45 | -2.36 |
| BTC Hourly | lstm | LSTM | 865 | 368 | 497 | 42.54% | 38.33% | 42.08% | 7.46 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 865 | 364 | 501 | 42.08% | 40.42% | 42.92% | 7.92 pp | -137 | 46 | -2.98 |
| Consolidated Market Hours | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 16 | 5 | 11 | 31.25% | 31.25% | 31.25% | 18.75 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 698 | 277 | 421 | 39.68% | 35.83% | 39.38% | 10.32 pp | -144 | 41 | -3.51 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 865 | 408 | 457 | 47.17% | 45.83% | 47.29% | 2.83 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 865 | 407 | 458 | 47.05% | 47.50% | 47.08% | 2.95 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 865 | 389 | 476 | 44.97% | 45.42% | 43.96% | 5.03 pp | -87 | 46 | -1.89 |
| BTC Hourly | rf | RandomForest | 865 | 385 | 480 | 44.51% | 43.75% | 44.17% | 5.49 pp | -95 | 46 | -2.07 |
| BTC Hourly | lstm | LSTM | 865 | 368 | 497 | 42.54% | 38.33% | 42.08% | 7.46 pp | -129 | 46 | -2.80 |
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
