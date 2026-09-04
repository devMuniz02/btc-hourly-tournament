# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T01:32:04.556021+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1220 | 932 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1095 | 730 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 763 | 492 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 765 | 546 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 141 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 16:00:00+00:00 | 141 | 32 | 109 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 492 | 237 | 255 | 48.17% | 44.58% | 48.12% | 1.83 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 492 | 232 | 260 | 47.15% | 50.00% | 47.71% | 2.85 pp | -28 | 47 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 720 | 347 | 373 | 48.19% | 46.25% | 47.92% | 1.81 pp | -26 | 43 | -0.60 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 492 | 230 | 262 | 46.75% | 43.33% | 47.29% | 3.25 pp | -32 | 47 | -0.68 |
| BTC Daily | transformer | Transformer | 720 | 344 | 376 | 47.78% | 46.25% | 50.21% | 2.22 pp | -32 | 43 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 898 | 430 | 468 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 546 | 253 | 293 | 46.34% | 48.75% | 47.29% | 3.66 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 546 | 251 | 295 | 45.97% | 44.17% | 47.08% | 4.03 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 546 | 250 | 296 | 45.79% | 47.92% | 46.88% | 4.21 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 898 | 424 | 474 | 47.22% | 47.50% | 46.46% | 2.78 pp | -50 | 47 | -1.06 |
| BTC Daily | nn | NN | 720 | 334 | 386 | 46.39% | 44.17% | 47.92% | 3.61 pp | -52 | 43 | -1.21 |
| BTC Market Hours | lstm | LSTM | 492 | 212 | 280 | 43.09% | 41.25% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 492 | 212 | 280 | 43.09% | 43.33% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 492 | 201 | 291 | 40.85% | 40.00% | 40.83% | 9.15 pp | -90 | 47 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 546 | 227 | 319 | 41.58% | 41.67% | 41.25% | 8.42 pp | -92 | 47 | -1.96 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| BTC Hourly | nn | NN | 898 | 400 | 498 | 44.54% | 44.58% | 42.50% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 898 | 400 | 498 | 44.54% | 45.00% | 44.38% | 5.46 pp | -98 | 47 | -2.09 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | lstm | LSTM | 720 | 312 | 408 | 43.33% | 37.92% | 42.08% | 6.67 pp | -96 | 43 | -2.23 |
| BTC Market Hours Daily | lstm | LSTM | 546 | 220 | 326 | 40.29% | 37.92% | 40.83% | 9.71 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 546 | 219 | 327 | 40.11% | 40.83% | 39.58% | 9.89 pp | -108 | 47 | -2.30 |
| BTC Daily | rf | RandomForest | 720 | 307 | 413 | 42.64% | 40.42% | 43.54% | 7.36 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 898 | 384 | 514 | 42.76% | 39.58% | 42.50% | 7.24 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 898 | 378 | 520 | 42.09% | 42.50% | 42.08% | 7.91 pp | -142 | 47 | -3.02 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 730 | 289 | 441 | 39.59% | 35.83% | 38.54% | 10.41 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 898 | 430 | 468 | 47.88% | 51.25% | 48.75% | 2.12 pp | -38 | 47 | -0.81 |
| BTC Hourly | transformer | Transformer | 898 | 424 | 474 | 47.22% | 47.50% | 46.46% | 2.78 pp | -50 | 47 | -1.06 |
| BTC Hourly | nn | NN | 898 | 400 | 498 | 44.54% | 44.58% | 42.50% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | rf | RandomForest | 898 | 400 | 498 | 44.54% | 45.00% | 44.38% | 5.46 pp | -98 | 47 | -2.09 |
| BTC Hourly | lstm | LSTM | 898 | 384 | 514 | 42.76% | 39.58% | 42.50% | 7.24 pp | -130 | 47 | -2.77 |
| BTC Hourly | xgb | XGBoost | 898 | 378 | 520 | 42.09% | 42.50% | 42.08% | 7.91 pp | -142 | 47 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 720 | 347 | 373 | 48.19% | 46.25% | 47.92% | 1.81 pp | -26 | 43 | -0.60 |
| BTC Daily | transformer | Transformer | 720 | 344 | 376 | 47.78% | 46.25% | 50.21% | 2.22 pp | -32 | 43 | -0.74 |
| BTC Daily | nn | NN | 720 | 334 | 386 | 46.39% | 44.17% | 47.92% | 3.61 pp | -52 | 43 | -1.21 |
| BTC Daily | lstm | LSTM | 720 | 312 | 408 | 43.33% | 37.92% | 42.08% | 6.67 pp | -96 | 43 | -2.23 |
| BTC Daily | rf | RandomForest | 720 | 307 | 413 | 42.64% | 40.42% | 43.54% | 7.36 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 730 | 289 | 441 | 39.59% | 35.83% | 38.54% | 10.41 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 492 | 237 | 255 | 48.17% | 44.58% | 48.12% | 1.83 pp | -18 | 47 | -0.38 |
| BTC Market Hours | nn | NN | 492 | 232 | 260 | 47.15% | 50.00% | 47.71% | 2.85 pp | -28 | 47 | -0.60 |
| BTC Market Hours | transformer | Transformer | 492 | 230 | 262 | 46.75% | 43.33% | 47.29% | 3.25 pp | -32 | 47 | -0.68 |
| BTC Market Hours | lstm | LSTM | 492 | 212 | 280 | 43.09% | 41.25% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | rf | RandomForest | 492 | 212 | 280 | 43.09% | 43.33% | 43.33% | 6.91 pp | -68 | 47 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 492 | 201 | 291 | 40.85% | 40.00% | 40.83% | 9.15 pp | -90 | 47 | -1.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 546 | 253 | 293 | 46.34% | 48.75% | 47.29% | 3.66 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | nn | NN | 546 | 251 | 295 | 45.97% | 44.17% | 47.08% | 4.03 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 546 | 250 | 296 | 45.79% | 47.92% | 46.88% | 4.21 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 546 | 227 | 319 | 41.58% | 41.67% | 41.25% | 8.42 pp | -92 | 47 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 546 | 220 | 326 | 40.29% | 37.92% | 40.83% | 9.71 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 546 | 219 | 327 | 40.11% | 40.83% | 39.58% | 9.89 pp | -108 | 47 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 74 | 67 | 52.48% | 52.48% | 52.48% | 2.48 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
