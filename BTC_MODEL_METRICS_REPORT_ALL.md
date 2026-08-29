# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T02:17:55.464216+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 830 | 289 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 994 | 629 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 584 | 391 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 586 | 445 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 1 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 00:00:00+00:00 | 50 | 1 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 391 | 193 | 198 | 49.36% | 48.33% | 49.36% | 0.64 pp | -5 | 39 | -0.13 |
| BTC Daily | transformer | Transformer | 619 | 304 | 315 | 49.11% | 48.33% | 50.21% | 0.89 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 619 | 302 | 317 | 48.79% | 47.08% | 50.21% | 1.21 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 391 | 183 | 208 | 46.80% | 49.17% | 46.80% | 3.20 pp | -25 | 39 | -0.64 |
| BTC Market Hours | transformer | Transformer | 391 | 183 | 208 | 46.80% | 44.58% | 46.80% | 3.20 pp | -25 | 39 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 445 | 207 | 238 | 46.52% | 48.75% | 46.52% | 3.48 pp | -31 | 39 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 445 | 205 | 240 | 46.07% | 46.25% | 46.07% | 3.93 pp | -35 | 39 | -0.90 |
| BTC Daily | nn | NN | 619 | 291 | 328 | 47.01% | 43.33% | 49.17% | 2.99 pp | -37 | 39 | -0.95 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 796 | 376 | 420 | 47.24% | 45.00% | 46.67% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Market Hours Daily | nn | NN | 445 | 202 | 243 | 45.39% | 46.25% | 45.39% | 4.61 pp | -41 | 39 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 796 | 375 | 421 | 47.11% | 45.00% | 46.88% | 2.89 pp | -46 | 43 | -1.07 |
| BTC Market Hours | lstm | LSTM | 391 | 170 | 221 | 43.48% | 43.33% | 43.48% | 6.52 pp | -51 | 39 | -1.31 |
| BTC Market Hours | rf | RandomForest | 391 | 167 | 224 | 42.71% | 40.83% | 42.71% | 7.29 pp | -57 | 39 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 619 | 276 | 343 | 44.59% | 43.33% | 44.58% | 5.41 pp | -67 | 39 | -1.72 |
| BTC Hourly | nn | NN | 796 | 359 | 437 | 45.10% | 41.25% | 45.42% | 4.90 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 391 | 160 | 231 | 40.92% | 39.17% | 40.92% | 9.08 pp | -71 | 39 | -1.82 |
| BTC Hourly | rf | RandomForest | 796 | 355 | 441 | 44.60% | 43.33% | 44.17% | 5.40 pp | -86 | 43 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 445 | 182 | 263 | 40.90% | 40.00% | 40.90% | 9.10 pp | -81 | 39 | -2.08 |
| BTC Hourly | lstm | LSTM | 796 | 352 | 444 | 44.22% | 45.00% | 45.83% | 5.78 pp | -92 | 43 | -2.14 |
| BTC Daily | rf | RandomForest | 619 | 265 | 354 | 42.81% | 42.50% | 43.75% | 7.19 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | lstm | LSTM | 445 | 178 | 267 | 40.00% | 38.33% | 40.00% | 10.00 pp | -89 | 39 | -2.28 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 445 | 176 | 269 | 39.55% | 37.92% | 39.55% | 10.45 pp | -93 | 39 | -2.38 |
| BTC Hourly | xgb | XGBoost | 796 | 340 | 456 | 42.71% | 40.00% | 44.38% | 7.29 pp | -116 | 43 | -2.70 |
| BTC Daily | xgb | XGBoost | 629 | 250 | 379 | 39.75% | 33.33% | 40.21% | 10.25 pp | -129 | 39 | -3.31 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 796 | 376 | 420 | 47.24% | 45.00% | 46.67% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 796 | 375 | 421 | 47.11% | 45.00% | 46.88% | 2.89 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 796 | 359 | 437 | 45.10% | 41.25% | 45.42% | 4.90 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 796 | 355 | 441 | 44.60% | 43.33% | 44.17% | 5.40 pp | -86 | 43 | -2.00 |
| BTC Hourly | lstm | LSTM | 796 | 352 | 444 | 44.22% | 45.00% | 45.83% | 5.78 pp | -92 | 43 | -2.14 |
| BTC Hourly | xgb | XGBoost | 796 | 340 | 456 | 42.71% | 40.00% | 44.38% | 7.29 pp | -116 | 43 | -2.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 619 | 304 | 315 | 49.11% | 48.33% | 50.21% | 0.89 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 619 | 302 | 317 | 48.79% | 47.08% | 50.21% | 1.21 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 619 | 291 | 328 | 47.01% | 43.33% | 49.17% | 2.99 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 619 | 276 | 343 | 44.59% | 43.33% | 44.58% | 5.41 pp | -67 | 39 | -1.72 |
| BTC Daily | rf | RandomForest | 619 | 265 | 354 | 42.81% | 42.50% | 43.75% | 7.19 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 629 | 250 | 379 | 39.75% | 33.33% | 40.21% | 10.25 pp | -129 | 39 | -3.31 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 391 | 193 | 198 | 49.36% | 48.33% | 49.36% | 0.64 pp | -5 | 39 | -0.13 |
| BTC Market Hours | nn | NN | 391 | 183 | 208 | 46.80% | 49.17% | 46.80% | 3.20 pp | -25 | 39 | -0.64 |
| BTC Market Hours | transformer | Transformer | 391 | 183 | 208 | 46.80% | 44.58% | 46.80% | 3.20 pp | -25 | 39 | -0.64 |
| BTC Market Hours | lstm | LSTM | 391 | 170 | 221 | 43.48% | 43.33% | 43.48% | 6.52 pp | -51 | 39 | -1.31 |
| BTC Market Hours | rf | RandomForest | 391 | 167 | 224 | 42.71% | 40.83% | 42.71% | 7.29 pp | -57 | 39 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 391 | 160 | 231 | 40.92% | 39.17% | 40.92% | 9.08 pp | -71 | 39 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 445 | 207 | 238 | 46.52% | 48.75% | 46.52% | 3.48 pp | -31 | 39 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 445 | 205 | 240 | 46.07% | 46.25% | 46.07% | 3.93 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | nn | NN | 445 | 202 | 243 | 45.39% | 46.25% | 45.39% | 4.61 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 445 | 182 | 263 | 40.90% | 40.00% | 40.90% | 9.10 pp | -81 | 39 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 445 | 178 | 267 | 40.00% | 38.33% | 40.00% | 10.00 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 445 | 176 | 269 | 39.55% | 37.92% | 39.55% | 10.45 pp | -93 | 39 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 26 | 24 | 52.00% | 52.00% | 52.00% | 2.00 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
