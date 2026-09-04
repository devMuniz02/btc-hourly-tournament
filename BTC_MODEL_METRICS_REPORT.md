# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T03:02:24.350785+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1221 | 933 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1096 | 731 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 764 | 493 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 766 | 547 | 217 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 493 | 237 | 256 | 48.07% | 44.17% | 47.92% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 493 | 233 | 260 | 47.26% | 50.42% | 47.71% | 2.74 pp | -27 | 47 | -0.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 721 | 347 | 374 | 48.13% | 46.25% | 47.71% | 1.87 pp | -27 | 43 | -0.63 |
| Consolidated Hourly | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 67 | 74 | 47.52% | 47.52% | 47.52% | 2.48 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 493 | 230 | 263 | 46.65% | 42.92% | 47.29% | 3.35 pp | -33 | 47 | -0.70 |
| BTC Daily | transformer | Transformer | 721 | 345 | 376 | 47.85% | 46.67% | 50.21% | 2.15 pp | -31 | 43 | -0.72 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 547 | 254 | 293 | 46.44% | 49.17% | 47.29% | 3.56 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 547 | 252 | 295 | 46.07% | 44.58% | 47.08% | 3.93 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 547 | 251 | 296 | 45.89% | 48.33% | 46.88% | 4.11 pp | -45 | 47 | -0.96 |
| Consolidated Hourly | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Daily | nn | NN | 721 | 334 | 387 | 46.32% | 44.17% | 47.71% | 3.68 pp | -53 | 43 | -1.23 |
| BTC Market Hours | lstm | LSTM | 493 | 213 | 280 | 43.20% | 41.25% | 43.33% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 493 | 213 | 280 | 43.20% | 43.75% | 43.54% | 6.80 pp | -67 | 47 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 61 | 80 | 43.26% | 43.26% | 43.26% | 6.74 pp | -19 | 11 | -1.73 |
| BTC Market Hours | xgb | XGBoost | 493 | 202 | 291 | 40.97% | 40.42% | 41.04% | 9.03 pp | -89 | 47 | -1.89 |
| BTC Market Hours Daily | rf | RandomForest | 547 | 227 | 320 | 41.50% | 41.67% | 41.25% | 8.50 pp | -93 | 47 | -1.98 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Daily | lstm | LSTM | 721 | 312 | 409 | 43.27% | 37.92% | 42.08% | 6.73 pp | -97 | 43 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 547 | 220 | 327 | 40.22% | 37.92% | 40.62% | 9.78 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 547 | 219 | 328 | 40.04% | 40.83% | 39.38% | 9.96 pp | -109 | 47 | -2.32 |
| BTC Daily | rf | RandomForest | 721 | 308 | 413 | 42.72% | 40.83% | 43.54% | 7.28 pp | -105 | 43 | -2.44 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 731 | 290 | 441 | 39.67% | 36.25% | 38.54% | 10.33 pp | -151 | 43 | -3.51 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 899 | 430 | 469 | 47.83% | 51.25% | 48.54% | 2.17 pp | -39 | 47 | -0.83 |
| BTC Hourly | transformer | Transformer | 899 | 425 | 474 | 47.27% | 47.92% | 46.67% | 2.73 pp | -49 | 47 | -1.04 |
| BTC Hourly | nn | NN | 899 | 400 | 499 | 44.49% | 44.17% | 42.29% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | rf | RandomForest | 899 | 400 | 499 | 44.49% | 45.00% | 44.17% | 5.51 pp | -99 | 47 | -2.11 |
| BTC Hourly | lstm | LSTM | 899 | 384 | 515 | 42.71% | 39.17% | 42.29% | 7.29 pp | -131 | 47 | -2.79 |
| BTC Hourly | xgb | XGBoost | 899 | 378 | 521 | 42.05% | 42.50% | 41.88% | 7.95 pp | -143 | 47 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 721 | 347 | 374 | 48.13% | 46.25% | 47.71% | 1.87 pp | -27 | 43 | -0.63 |
| BTC Daily | transformer | Transformer | 721 | 345 | 376 | 47.85% | 46.67% | 50.21% | 2.15 pp | -31 | 43 | -0.72 |
| BTC Daily | nn | NN | 721 | 334 | 387 | 46.32% | 44.17% | 47.71% | 3.68 pp | -53 | 43 | -1.23 |
| BTC Daily | lstm | LSTM | 721 | 312 | 409 | 43.27% | 37.92% | 42.08% | 6.73 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 721 | 308 | 413 | 42.72% | 40.83% | 43.54% | 7.28 pp | -105 | 43 | -2.44 |
| BTC Daily | xgb | XGBoost | 731 | 290 | 441 | 39.67% | 36.25% | 38.54% | 10.33 pp | -151 | 43 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 493 | 237 | 256 | 48.07% | 44.17% | 47.92% | 1.93 pp | -19 | 47 | -0.40 |
| BTC Market Hours | nn | NN | 493 | 233 | 260 | 47.26% | 50.42% | 47.71% | 2.74 pp | -27 | 47 | -0.57 |
| BTC Market Hours | transformer | Transformer | 493 | 230 | 263 | 46.65% | 42.92% | 47.29% | 3.35 pp | -33 | 47 | -0.70 |
| BTC Market Hours | lstm | LSTM | 493 | 213 | 280 | 43.20% | 41.25% | 43.33% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | rf | RandomForest | 493 | 213 | 280 | 43.20% | 43.75% | 43.54% | 6.80 pp | -67 | 47 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 493 | 202 | 291 | 40.97% | 40.42% | 41.04% | 9.03 pp | -89 | 47 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 547 | 254 | 293 | 46.44% | 49.17% | 47.29% | 3.56 pp | -39 | 47 | -0.83 |
| BTC Market Hours Daily | nn | NN | 547 | 252 | 295 | 46.07% | 44.58% | 47.08% | 3.93 pp | -43 | 47 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 547 | 251 | 296 | 45.89% | 48.33% | 46.88% | 4.11 pp | -45 | 47 | -0.96 |
| BTC Market Hours Daily | rf | RandomForest | 547 | 227 | 320 | 41.50% | 41.67% | 41.25% | 8.50 pp | -93 | 47 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 547 | 220 | 327 | 40.22% | 37.92% | 40.62% | 9.78 pp | -107 | 47 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 547 | 219 | 328 | 40.04% | 40.83% | 39.38% | 9.96 pp | -109 | 47 | -2.32 |

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
