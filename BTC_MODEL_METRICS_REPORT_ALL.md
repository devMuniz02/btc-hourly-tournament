# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T19:20:31.606087+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 793 | 326 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 957 | 592 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 18:00:00+00:00 | 510 | 354 | 155 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 18:00:00+00:00 | 512 | 408 | 102 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 13:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 13:00:00+00:00 | 19 | 19 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 13:00:00+00:00 | 19 | 1 | 18 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 13:00:00+00:00 | 19 | 1 | 18 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 354 | 175 | 179 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | transformer | Transformer | 582 | 288 | 294 | 49.48% | 52.08% | 49.79% | 0.52 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 582 | 287 | 295 | 49.31% | 49.17% | 49.79% | 0.69 pp | -8 | 37 | -0.22 |
| Consolidated Hourly | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 354 | 167 | 187 | 47.18% | 46.25% | 47.18% | 2.82 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 354 | 162 | 192 | 45.76% | 47.92% | 45.76% | 4.24 pp | -30 | 37 | -0.81 |
| BTC Daily | nn | NN | 582 | 275 | 307 | 47.25% | 45.42% | 48.12% | 2.75 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 408 | 187 | 221 | 45.83% | 46.25% | 45.83% | 4.17 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 408 | 187 | 221 | 45.83% | 47.08% | 45.83% | 4.17 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 408 | 186 | 222 | 45.59% | 47.50% | 45.59% | 4.41 pp | -36 | 37 | -0.97 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 759 | 356 | 403 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 759 | 355 | 404 | 46.77% | 44.17% | 45.42% | 3.23 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Daily | lstm | LSTM | 582 | 262 | 320 | 45.02% | 45.83% | 45.00% | 4.98 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 354 | 147 | 207 | 41.53% | 42.08% | 41.53% | 8.47 pp | -60 | 37 | -1.62 |
| BTC Hourly | rf | RandomForest | 759 | 340 | 419 | 44.80% | 44.58% | 44.38% | 5.20 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 759 | 339 | 420 | 44.66% | 41.25% | 45.21% | 5.34 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 408 | 168 | 240 | 41.18% | 40.00% | 41.18% | 8.82 pp | -72 | 37 | -1.95 |
| BTC Daily | rf | RandomForest | 582 | 252 | 330 | 43.30% | 44.58% | 43.96% | 6.70 pp | -78 | 37 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 408 | 164 | 244 | 40.20% | 38.75% | 40.20% | 9.80 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 408 | 164 | 244 | 40.20% | 37.92% | 40.20% | 9.80 pp | -80 | 37 | -2.16 |
| BTC Hourly | lstm | LSTM | 759 | 334 | 425 | 44.01% | 42.92% | 45.42% | 5.99 pp | -91 | 42 | -2.17 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |
| BTC Hourly | xgb | XGBoost | 759 | 326 | 433 | 42.95% | 41.67% | 44.38% | 7.05 pp | -107 | 42 | -2.55 |
| BTC Daily | xgb | XGBoost | 592 | 238 | 354 | 40.20% | 35.42% | 40.42% | 9.80 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 759 | 356 | 403 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 759 | 355 | 404 | 46.77% | 44.17% | 45.42% | 3.23 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 759 | 340 | 419 | 44.80% | 44.58% | 44.38% | 5.20 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 759 | 339 | 420 | 44.66% | 41.25% | 45.21% | 5.34 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 759 | 334 | 425 | 44.01% | 42.92% | 45.42% | 5.99 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 759 | 326 | 433 | 42.95% | 41.67% | 44.38% | 7.05 pp | -107 | 42 | -2.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 582 | 288 | 294 | 49.48% | 52.08% | 49.79% | 0.52 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 582 | 287 | 295 | 49.31% | 49.17% | 49.79% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | nn | NN | 582 | 275 | 307 | 47.25% | 45.42% | 48.12% | 2.75 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 582 | 262 | 320 | 45.02% | 45.83% | 45.00% | 4.98 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 582 | 252 | 330 | 43.30% | 44.58% | 43.96% | 6.70 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 592 | 238 | 354 | 40.20% | 35.42% | 40.42% | 9.80 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 354 | 175 | 179 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 354 | 167 | 187 | 47.18% | 46.25% | 47.18% | 2.82 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 354 | 162 | 192 | 45.76% | 47.92% | 45.76% | 4.24 pp | -30 | 37 | -0.81 |
| BTC Market Hours | lstm | LSTM | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | xgb | XGBoost | 354 | 147 | 207 | 41.53% | 42.08% | 41.53% | 8.47 pp | -60 | 37 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 408 | 187 | 221 | 45.83% | 46.25% | 45.83% | 4.17 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 408 | 187 | 221 | 45.83% | 47.08% | 45.83% | 4.17 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 408 | 186 | 222 | 45.59% | 47.50% | 45.59% | 4.41 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 408 | 168 | 240 | 41.18% | 40.00% | 41.18% | 8.82 pp | -72 | 37 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 408 | 164 | 244 | 40.20% | 38.75% | 40.20% | 9.80 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 408 | 164 | 244 | 40.20% | 37.92% | 40.20% | 9.80 pp | -80 | 37 | -2.16 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 19 | 12 | 7 | 63.16% | 63.16% | 63.16% | 13.16 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 19 | 6 | 13 | 31.58% | 31.58% | 31.58% | 18.42 pp | -7 | 3 | -2.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
