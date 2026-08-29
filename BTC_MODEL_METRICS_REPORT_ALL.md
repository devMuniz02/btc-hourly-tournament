# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T17:25:38.863737+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1130 | 842 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1006 | 641 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 16:00:00+00:00 | 601 | 403 | 197 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 16:00:00+00:00 | 603 | 457 | 144 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 23:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 23:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 23:00:00+00:00 | 61 | 1 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 23:00:00+00:00 | 61 | 1 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 403 | 200 | 203 | 49.63% | 48.75% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 6 | -0.17 |
| BTC Daily | transformer | Transformer | 631 | 309 | 322 | 48.97% | 47.50% | 49.38% | 1.03 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 631 | 308 | 323 | 48.81% | 46.67% | 50.21% | 1.19 pp | -15 | 39 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| BTC Market Hours | nn | NN | 403 | 191 | 212 | 47.39% | 50.83% | 47.39% | 2.61 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 403 | 186 | 217 | 46.15% | 42.50% | 46.15% | 3.85 pp | -31 | 40 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 457 | 212 | 245 | 46.39% | 46.67% | 46.39% | 3.61 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | nn | NN | 457 | 209 | 248 | 45.73% | 46.25% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 457 | 209 | 248 | 45.73% | 47.08% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Daily | nn | NN | 631 | 296 | 335 | 46.91% | 42.92% | 48.96% | 3.09 pp | -39 | 39 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 808 | 381 | 427 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 808 | 380 | 428 | 47.03% | 44.17% | 46.04% | 2.97 pp | -48 | 44 | -1.09 |
| BTC Market Hours | lstm | LSTM | 403 | 178 | 225 | 44.17% | 45.83% | 44.17% | 5.83 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 403 | 173 | 230 | 42.93% | 42.08% | 42.93% | 7.07 pp | -57 | 40 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| BTC Daily | lstm | LSTM | 631 | 279 | 352 | 44.22% | 42.08% | 43.54% | 5.78 pp | -73 | 39 | -1.87 |
| BTC Hourly | nn | NN | 808 | 362 | 446 | 44.80% | 40.00% | 44.38% | 5.20 pp | -84 | 44 | -1.91 |
| BTC Hourly | rf | RandomForest | 808 | 361 | 447 | 44.68% | 44.17% | 44.58% | 5.32 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 403 | 162 | 241 | 40.20% | 38.33% | 40.20% | 9.80 pp | -79 | 40 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 457 | 188 | 269 | 41.14% | 40.83% | 41.14% | 8.86 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 457 | 187 | 270 | 40.92% | 40.00% | 40.92% | 9.08 pp | -83 | 40 | -2.08 |
| BTC Hourly | lstm | LSTM | 808 | 354 | 454 | 43.81% | 42.50% | 44.79% | 6.19 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 631 | 269 | 362 | 42.63% | 42.08% | 43.54% | 7.37 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 457 | 180 | 277 | 39.39% | 37.08% | 39.39% | 10.61 pp | -97 | 40 | -2.42 |
| BTC Hourly | xgb | XGBoost | 808 | 343 | 465 | 42.45% | 40.00% | 43.12% | 7.55 pp | -122 | 44 | -2.77 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |
| BTC Daily | xgb | XGBoost | 641 | 251 | 390 | 39.16% | 31.25% | 39.38% | 10.84 pp | -139 | 39 | -3.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 808 | 381 | 427 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 808 | 380 | 428 | 47.03% | 44.17% | 46.04% | 2.97 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 808 | 362 | 446 | 44.80% | 40.00% | 44.38% | 5.20 pp | -84 | 44 | -1.91 |
| BTC Hourly | rf | RandomForest | 808 | 361 | 447 | 44.68% | 44.17% | 44.58% | 5.32 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 808 | 354 | 454 | 43.81% | 42.50% | 44.79% | 6.19 pp | -100 | 44 | -2.27 |
| BTC Hourly | xgb | XGBoost | 808 | 343 | 465 | 42.45% | 40.00% | 43.12% | 7.55 pp | -122 | 44 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 631 | 309 | 322 | 48.97% | 47.50% | 49.38% | 1.03 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 631 | 308 | 323 | 48.81% | 46.67% | 50.21% | 1.19 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 631 | 296 | 335 | 46.91% | 42.92% | 48.96% | 3.09 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 631 | 279 | 352 | 44.22% | 42.08% | 43.54% | 5.78 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 631 | 269 | 362 | 42.63% | 42.08% | 43.54% | 7.37 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 641 | 251 | 390 | 39.16% | 31.25% | 39.38% | 10.84 pp | -139 | 39 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 403 | 200 | 203 | 49.63% | 48.75% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| BTC Market Hours | nn | NN | 403 | 191 | 212 | 47.39% | 50.83% | 47.39% | 2.61 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 403 | 186 | 217 | 46.15% | 42.50% | 46.15% | 3.85 pp | -31 | 40 | -0.78 |
| BTC Market Hours | lstm | LSTM | 403 | 178 | 225 | 44.17% | 45.83% | 44.17% | 5.83 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 403 | 173 | 230 | 42.93% | 42.08% | 42.93% | 7.07 pp | -57 | 40 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 403 | 162 | 241 | 40.20% | 38.33% | 40.20% | 9.80 pp | -79 | 40 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 457 | 212 | 245 | 46.39% | 46.67% | 46.39% | 3.61 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | nn | NN | 457 | 209 | 248 | 45.73% | 46.25% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 457 | 209 | 248 | 45.73% | 47.08% | 45.73% | 4.27 pp | -39 | 40 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 457 | 188 | 269 | 41.14% | 40.83% | 41.14% | 8.86 pp | -81 | 40 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 457 | 187 | 270 | 40.92% | 40.00% | 40.92% | 9.08 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 457 | 180 | 277 | 39.39% | 37.08% | 39.39% | 10.61 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 35 | 26 | 57.38% | 57.38% | 57.38% | 7.38 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 6 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 6 | -1.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
