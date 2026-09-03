# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T15:25:34.233930+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1213 | 925 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1089 | 724 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 14:00:00+00:00 | 747 | 486 | 260 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 14:00:00+00:00 | 749 | 540 | 207 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 134 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 134 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 134 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T12:00:00+00:00 | 135 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 714 | 366 | 348 | 51.26% | 48.33% | 51.46% | 1.26 pp | 18 | 43 | 0.42 |
| Consolidated Hourly | rf | RandomForest | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | xgb | XGBoost | 134 | 66 | 68 | 49.25% | 49.25% | 49.25% | 0.75 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 134 | 66 | 68 | 49.25% | 49.25% | 49.25% | 0.75 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 486 | 234 | 252 | 48.15% | 43.75% | 48.12% | 1.85 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 486 | 229 | 257 | 47.12% | 48.75% | 47.29% | 2.88 pp | -28 | 47 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 486 | 227 | 259 | 46.71% | 42.50% | 47.08% | 3.29 pp | -32 | 47 | -0.68 |
| BTC Daily | nn | NN | 714 | 340 | 374 | 47.62% | 46.67% | 48.54% | 2.38 pp | -34 | 43 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 540 | 250 | 290 | 46.30% | 49.17% | 47.29% | 3.70 pp | -40 | 47 | -0.85 |
| Consolidated Hourly | lstm | LSTM | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 891 | 424 | 467 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Daily | transformer | Transformer | 714 | 336 | 378 | 47.06% | 45.42% | 48.96% | 2.94 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 540 | 247 | 293 | 45.74% | 47.50% | 46.88% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 540 | 247 | 293 | 45.74% | 44.17% | 46.67% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 891 | 422 | 469 | 47.36% | 48.33% | 47.29% | 2.64 pp | -47 | 47 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 11 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 486 | 209 | 277 | 43.00% | 41.25% | 42.92% | 7.00 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 486 | 208 | 278 | 42.80% | 42.08% | 43.12% | 7.20 pp | -70 | 47 | -1.49 |
| BTC Daily | lstm | LSTM | 714 | 322 | 392 | 45.10% | 38.33% | 44.38% | 4.90 pp | -70 | 43 | -1.63 |
| Consolidated Hourly | transformer | Transformer | 134 | 58 | 76 | 43.28% | 43.28% | 43.28% | 6.72 pp | -18 | 11 | -1.64 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 134 | 58 | 76 | 43.28% | 43.28% | 43.28% | 6.72 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 714 | 319 | 395 | 44.68% | 41.67% | 44.58% | 5.32 pp | -76 | 43 | -1.77 |
| BTC Hourly | nn | NN | 891 | 399 | 492 | 44.78% | 46.25% | 42.92% | 5.22 pp | -93 | 47 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 486 | 196 | 290 | 40.33% | 39.17% | 40.42% | 9.67 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 891 | 398 | 493 | 44.67% | 45.42% | 44.17% | 5.33 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 540 | 222 | 318 | 41.11% | 41.25% | 41.25% | 8.89 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 540 | 217 | 323 | 40.19% | 37.92% | 40.83% | 9.81 pp | -106 | 47 | -2.26 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 540 | 215 | 325 | 39.81% | 39.58% | 39.58% | 10.19 pp | -110 | 47 | -2.34 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 891 | 382 | 509 | 42.87% | 39.17% | 42.29% | 7.13 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 891 | 377 | 514 | 42.31% | 42.92% | 42.29% | 7.69 pp | -137 | 47 | -2.91 |
| BTC Daily | xgb | XGBoost | 724 | 291 | 433 | 40.19% | 36.67% | 39.38% | 9.81 pp | -142 | 43 | -3.30 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 891 | 424 | 467 | 47.59% | 49.58% | 47.92% | 2.41 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 891 | 422 | 469 | 47.36% | 48.33% | 47.29% | 2.64 pp | -47 | 47 | -1.00 |
| BTC Hourly | nn | NN | 891 | 399 | 492 | 44.78% | 46.25% | 42.92% | 5.22 pp | -93 | 47 | -1.98 |
| BTC Hourly | rf | RandomForest | 891 | 398 | 493 | 44.67% | 45.42% | 44.17% | 5.33 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 891 | 382 | 509 | 42.87% | 39.17% | 42.29% | 7.13 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 891 | 377 | 514 | 42.31% | 42.92% | 42.29% | 7.69 pp | -137 | 47 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 714 | 366 | 348 | 51.26% | 48.33% | 51.46% | 1.26 pp | 18 | 43 | 0.42 |
| BTC Daily | nn | NN | 714 | 340 | 374 | 47.62% | 46.67% | 48.54% | 2.38 pp | -34 | 43 | -0.79 |
| BTC Daily | transformer | Transformer | 714 | 336 | 378 | 47.06% | 45.42% | 48.96% | 2.94 pp | -42 | 43 | -0.98 |
| BTC Daily | lstm | LSTM | 714 | 322 | 392 | 45.10% | 38.33% | 44.38% | 4.90 pp | -70 | 43 | -1.63 |
| BTC Daily | rf | RandomForest | 714 | 319 | 395 | 44.68% | 41.67% | 44.58% | 5.32 pp | -76 | 43 | -1.77 |
| BTC Daily | xgb | XGBoost | 724 | 291 | 433 | 40.19% | 36.67% | 39.38% | 9.81 pp | -142 | 43 | -3.30 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 486 | 234 | 252 | 48.15% | 43.75% | 48.12% | 1.85 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 486 | 229 | 257 | 47.12% | 48.75% | 47.29% | 2.88 pp | -28 | 47 | -0.60 |
| BTC Market Hours | transformer | Transformer | 486 | 227 | 259 | 46.71% | 42.50% | 47.08% | 3.29 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 486 | 209 | 277 | 43.00% | 41.25% | 42.92% | 7.00 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 486 | 208 | 278 | 42.80% | 42.08% | 43.12% | 7.20 pp | -70 | 47 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 486 | 196 | 290 | 40.33% | 39.17% | 40.42% | 9.67 pp | -94 | 47 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 540 | 250 | 290 | 46.30% | 49.17% | 47.29% | 3.70 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 540 | 247 | 293 | 45.74% | 47.50% | 46.88% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 540 | 247 | 293 | 45.74% | 44.17% | 46.67% | 4.26 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 540 | 222 | 318 | 41.11% | 41.25% | 41.25% | 8.89 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 540 | 217 | 323 | 40.19% | 37.92% | 40.83% | 9.81 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 540 | 215 | 325 | 39.81% | 39.58% | 39.58% | 10.19 pp | -110 | 47 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 11 | 0.18 |
| Consolidated Hourly | xgb | XGBoost | 134 | 66 | 68 | 49.25% | 49.25% | 49.25% | 0.75 pp | -2 | 11 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| Consolidated Hourly | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 134 | 58 | 76 | 43.28% | 43.28% | 43.28% | 6.72 pp | -18 | 11 | -1.64 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 134 | 68 | 66 | 50.75% | 50.75% | 50.75% | 0.75 pp | 2 | 11 | 0.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 134 | 66 | 68 | 49.25% | 49.25% | 49.25% | 0.75 pp | -2 | 11 | -0.18 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 134 | 65 | 69 | 48.51% | 48.51% | 48.51% | 1.49 pp | -4 | 11 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 134 | 62 | 72 | 46.27% | 46.27% | 46.27% | 3.73 pp | -10 | 11 | -0.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 134 | 60 | 74 | 44.78% | 44.78% | 44.78% | 5.22 pp | -14 | 11 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 134 | 58 | 76 | 43.28% | 43.28% | 43.28% | 6.72 pp | -18 | 11 | -1.64 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 29 | 11 | 18 | 37.93% | 37.93% | 37.93% | 12.07 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 29 | 9 | 20 | 31.03% | 31.03% | 31.03% | 18.97 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
