# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T00:18:51.506088+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1170 | 882 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1046 | 681 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 23:00:00+00:00 | 674 | 443 | 230 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 23:00:00+00:00 | 676 | 497 | 177 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 96 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 443 | 216 | 227 | 48.76% | 45.00% | 48.76% | 1.24 pp | -11 | 43 | -0.26 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 671 | 328 | 343 | 48.88% | 47.50% | 49.58% | 1.12 pp | -15 | 41 | -0.37 |
| BTC Market Hours | nn | NN | 443 | 210 | 233 | 47.40% | 48.75% | 47.40% | 2.60 pp | -23 | 43 | -0.53 |
| BTC Daily | transformer | Transformer | 671 | 324 | 347 | 48.29% | 45.42% | 49.38% | 1.71 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 443 | 202 | 241 | 45.60% | 40.00% | 45.60% | 4.40 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 497 | 228 | 269 | 45.88% | 46.25% | 46.46% | 4.12 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | nn | NN | 497 | 228 | 269 | 45.88% | 43.75% | 46.67% | 4.12 pp | -41 | 43 | -0.95 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 848 | 401 | 447 | 47.29% | 47.50% | 47.08% | 2.71 pp | -46 | 45 | -1.02 |
| BTC Daily | nn | NN | 671 | 314 | 357 | 46.80% | 43.33% | 49.17% | 3.20 pp | -43 | 41 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 497 | 224 | 273 | 45.07% | 45.00% | 45.00% | 4.93 pp | -49 | 43 | -1.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 848 | 398 | 450 | 46.93% | 44.58% | 46.67% | 3.07 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 443 | 191 | 252 | 43.12% | 43.33% | 43.12% | 6.88 pp | -61 | 43 | -1.42 |
| BTC Market Hours | lstm | LSTM | 443 | 190 | 253 | 42.89% | 41.25% | 42.89% | 7.11 pp | -63 | 43 | -1.47 |
| BTC Hourly | nn | NN | 848 | 383 | 465 | 45.17% | 44.17% | 44.58% | 4.83 pp | -82 | 45 | -1.82 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 497 | 205 | 292 | 41.25% | 41.25% | 41.46% | 8.75 pp | -87 | 43 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 443 | 178 | 265 | 40.18% | 38.33% | 40.18% | 9.82 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 671 | 293 | 378 | 43.67% | 38.33% | 42.92% | 6.33 pp | -85 | 41 | -2.07 |
| BTC Hourly | rf | RandomForest | 848 | 377 | 471 | 44.46% | 42.92% | 43.75% | 5.54 pp | -94 | 45 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 497 | 198 | 299 | 39.84% | 37.92% | 40.42% | 10.16 pp | -101 | 43 | -2.35 |
| BTC Daily | rf | RandomForest | 671 | 287 | 384 | 42.77% | 40.83% | 43.75% | 7.23 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 497 | 195 | 302 | 39.24% | 36.25% | 39.17% | 10.76 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 848 | 363 | 485 | 42.81% | 39.17% | 42.29% | 7.19 pp | -122 | 45 | -2.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Hourly | xgb | XGBoost | 848 | 356 | 492 | 41.98% | 39.58% | 42.29% | 8.02 pp | -136 | 45 | -3.02 |
| BTC Daily | xgb | XGBoost | 681 | 271 | 410 | 39.79% | 35.00% | 39.58% | 10.21 pp | -139 | 41 | -3.39 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 848 | 401 | 447 | 47.29% | 47.50% | 47.08% | 2.71 pp | -46 | 45 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 848 | 398 | 450 | 46.93% | 44.58% | 46.67% | 3.07 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 848 | 383 | 465 | 45.17% | 44.17% | 44.58% | 4.83 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 848 | 377 | 471 | 44.46% | 42.92% | 43.75% | 5.54 pp | -94 | 45 | -2.09 |
| BTC Hourly | lstm | LSTM | 848 | 363 | 485 | 42.81% | 39.17% | 42.29% | 7.19 pp | -122 | 45 | -2.71 |
| BTC Hourly | xgb | XGBoost | 848 | 356 | 492 | 41.98% | 39.58% | 42.29% | 8.02 pp | -136 | 45 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 671 | 328 | 343 | 48.88% | 47.50% | 49.58% | 1.12 pp | -15 | 41 | -0.37 |
| BTC Daily | transformer | Transformer | 671 | 324 | 347 | 48.29% | 45.42% | 49.38% | 1.71 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 671 | 314 | 357 | 46.80% | 43.33% | 49.17% | 3.20 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 671 | 293 | 378 | 43.67% | 38.33% | 42.92% | 6.33 pp | -85 | 41 | -2.07 |
| BTC Daily | rf | RandomForest | 671 | 287 | 384 | 42.77% | 40.83% | 43.75% | 7.23 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 681 | 271 | 410 | 39.79% | 35.00% | 39.58% | 10.21 pp | -139 | 41 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 443 | 216 | 227 | 48.76% | 45.00% | 48.76% | 1.24 pp | -11 | 43 | -0.26 |
| BTC Market Hours | nn | NN | 443 | 210 | 233 | 47.40% | 48.75% | 47.40% | 2.60 pp | -23 | 43 | -0.53 |
| BTC Market Hours | transformer | Transformer | 443 | 202 | 241 | 45.60% | 40.00% | 45.60% | 4.40 pp | -39 | 43 | -0.91 |
| BTC Market Hours | rf | RandomForest | 443 | 191 | 252 | 43.12% | 43.33% | 43.12% | 6.88 pp | -61 | 43 | -1.42 |
| BTC Market Hours | lstm | LSTM | 443 | 190 | 253 | 42.89% | 41.25% | 42.89% | 7.11 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 443 | 178 | 265 | 40.18% | 38.33% | 40.18% | 9.82 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 497 | 228 | 269 | 45.88% | 46.25% | 46.46% | 4.12 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | nn | NN | 497 | 228 | 269 | 45.88% | 43.75% | 46.67% | 4.12 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 497 | 224 | 273 | 45.07% | 45.00% | 45.00% | 4.93 pp | -49 | 43 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 497 | 205 | 292 | 41.25% | 41.25% | 41.46% | 8.75 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 497 | 198 | 299 | 39.84% | 37.92% | 40.42% | 10.16 pp | -101 | 43 | -2.35 |
| BTC Market Hours Daily | xgb | XGBoost | 497 | 195 | 302 | 39.24% | 36.25% | 39.17% | 10.76 pp | -107 | 43 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
