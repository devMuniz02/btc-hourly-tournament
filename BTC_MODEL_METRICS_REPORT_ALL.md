# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T09:13:53.228854+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1209 | 921 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1085 | 720 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 740 | 482 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 742 | 536 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 710 | 366 | 344 | 51.55% | 49.58% | 51.67% | 1.55 pp | 22 | 42 | 0.52 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 482 | 232 | 250 | 48.13% | 44.17% | 48.33% | 1.87 pp | -18 | 46 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| BTC Market Hours | nn | NN | 482 | 228 | 254 | 47.30% | 49.17% | 47.50% | 2.70 pp | -26 | 46 | -0.57 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| BTC Market Hours | transformer | Transformer | 482 | 225 | 257 | 46.68% | 42.08% | 46.88% | 3.32 pp | -32 | 46 | -0.70 |
| BTC Daily | nn | NN | 710 | 339 | 371 | 47.75% | 47.08% | 48.75% | 2.25 pp | -32 | 42 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 536 | 249 | 287 | 46.46% | 49.58% | 47.50% | 3.54 pp | -38 | 46 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 887 | 422 | 465 | 47.58% | 49.58% | 48.12% | 2.42 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 887 | 422 | 465 | 47.58% | 48.75% | 47.92% | 2.42 pp | -43 | 47 | -0.91 |
| BTC Daily | transformer | Transformer | 710 | 335 | 375 | 47.18% | 45.83% | 49.17% | 2.82 pp | -40 | 42 | -0.95 |
| BTC Market Hours Daily | nn | NN | 536 | 246 | 290 | 45.90% | 44.17% | 46.88% | 4.10 pp | -44 | 46 | -0.96 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 536 | 245 | 291 | 45.71% | 47.92% | 46.67% | 4.29 pp | -46 | 46 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| BTC Market Hours | lstm | LSTM | 482 | 208 | 274 | 43.15% | 42.08% | 43.12% | 6.85 pp | -66 | 46 | -1.43 |
| BTC Market Hours | rf | RandomForest | 482 | 206 | 276 | 42.74% | 42.50% | 42.92% | 7.26 pp | -70 | 46 | -1.52 |
| BTC Daily | lstm | LSTM | 710 | 321 | 389 | 45.21% | 39.58% | 44.58% | 4.79 pp | -68 | 42 | -1.62 |
| BTC Daily | rf | RandomForest | 710 | 318 | 392 | 44.79% | 42.08% | 44.79% | 5.21 pp | -74 | 42 | -1.76 |
| BTC Hourly | nn | NN | 887 | 397 | 490 | 44.76% | 45.83% | 43.12% | 5.24 pp | -93 | 47 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 482 | 195 | 287 | 40.46% | 39.58% | 40.62% | 9.54 pp | -92 | 46 | -2.00 |
| BTC Hourly | rf | RandomForest | 887 | 396 | 491 | 44.64% | 44.58% | 44.38% | 5.36 pp | -95 | 47 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 536 | 220 | 316 | 41.04% | 41.25% | 41.25% | 8.96 pp | -96 | 46 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 536 | 213 | 323 | 39.74% | 37.50% | 40.21% | 10.26 pp | -110 | 46 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 536 | 213 | 323 | 39.74% | 39.17% | 39.38% | 10.26 pp | -110 | 46 | -2.39 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| BTC Hourly | lstm | LSTM | 887 | 380 | 507 | 42.84% | 38.75% | 42.29% | 7.16 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 887 | 375 | 512 | 42.28% | 42.08% | 42.29% | 7.72 pp | -137 | 47 | -2.91 |
| BTC Daily | xgb | XGBoost | 720 | 289 | 431 | 40.14% | 36.67% | 39.79% | 9.86 pp | -142 | 42 | -3.38 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 887 | 422 | 465 | 47.58% | 49.58% | 48.12% | 2.42 pp | -43 | 47 | -0.91 |
| BTC Hourly | transformer | Transformer | 887 | 422 | 465 | 47.58% | 48.75% | 47.92% | 2.42 pp | -43 | 47 | -0.91 |
| BTC Hourly | nn | NN | 887 | 397 | 490 | 44.76% | 45.83% | 43.12% | 5.24 pp | -93 | 47 | -1.98 |
| BTC Hourly | rf | RandomForest | 887 | 396 | 491 | 44.64% | 44.58% | 44.38% | 5.36 pp | -95 | 47 | -2.02 |
| BTC Hourly | lstm | LSTM | 887 | 380 | 507 | 42.84% | 38.75% | 42.29% | 7.16 pp | -127 | 47 | -2.70 |
| BTC Hourly | xgb | XGBoost | 887 | 375 | 512 | 42.28% | 42.08% | 42.29% | 7.72 pp | -137 | 47 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 710 | 366 | 344 | 51.55% | 49.58% | 51.67% | 1.55 pp | 22 | 42 | 0.52 |
| BTC Daily | nn | NN | 710 | 339 | 371 | 47.75% | 47.08% | 48.75% | 2.25 pp | -32 | 42 | -0.76 |
| BTC Daily | transformer | Transformer | 710 | 335 | 375 | 47.18% | 45.83% | 49.17% | 2.82 pp | -40 | 42 | -0.95 |
| BTC Daily | lstm | LSTM | 710 | 321 | 389 | 45.21% | 39.58% | 44.58% | 4.79 pp | -68 | 42 | -1.62 |
| BTC Daily | rf | RandomForest | 710 | 318 | 392 | 44.79% | 42.08% | 44.79% | 5.21 pp | -74 | 42 | -1.76 |
| BTC Daily | xgb | XGBoost | 720 | 289 | 431 | 40.14% | 36.67% | 39.79% | 9.86 pp | -142 | 42 | -3.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 482 | 232 | 250 | 48.13% | 44.17% | 48.33% | 1.87 pp | -18 | 46 | -0.39 |
| BTC Market Hours | nn | NN | 482 | 228 | 254 | 47.30% | 49.17% | 47.50% | 2.70 pp | -26 | 46 | -0.57 |
| BTC Market Hours | transformer | Transformer | 482 | 225 | 257 | 46.68% | 42.08% | 46.88% | 3.32 pp | -32 | 46 | -0.70 |
| BTC Market Hours | lstm | LSTM | 482 | 208 | 274 | 43.15% | 42.08% | 43.12% | 6.85 pp | -66 | 46 | -1.43 |
| BTC Market Hours | rf | RandomForest | 482 | 206 | 276 | 42.74% | 42.50% | 42.92% | 7.26 pp | -70 | 46 | -1.52 |
| BTC Market Hours | xgb | XGBoost | 482 | 195 | 287 | 40.46% | 39.58% | 40.62% | 9.54 pp | -92 | 46 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 536 | 249 | 287 | 46.46% | 49.58% | 47.50% | 3.54 pp | -38 | 46 | -0.83 |
| BTC Market Hours Daily | nn | NN | 536 | 246 | 290 | 45.90% | 44.17% | 46.88% | 4.10 pp | -44 | 46 | -0.96 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 536 | 245 | 291 | 45.71% | 47.92% | 46.67% | 4.29 pp | -46 | 46 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 536 | 220 | 316 | 41.04% | 41.25% | 41.25% | 8.96 pp | -96 | 46 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 536 | 213 | 323 | 39.74% | 37.50% | 40.21% | 10.26 pp | -110 | 46 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 536 | 213 | 323 | 39.74% | 39.17% | 39.38% | 10.26 pp | -110 | 46 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
