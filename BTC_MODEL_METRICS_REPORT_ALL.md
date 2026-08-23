# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T15:52:44.741112+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 14:00:00+00:00 | 1118 | 790 | 328 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 14:00:00+00:00 | 954 | 590 | 363 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 14:00:00+00:00 | 504 | 352 | 151 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 14:00:00+00:00 | 506 | 406 | 98 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 00:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 00:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 00:00:00+00:00 | 17 | 1 | 16 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 00:00:00+00:00 | 17 | 1 | 16 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 352 | 173 | 179 | 49.15% | 47.08% | 49.15% | 0.85 pp | -6 | 36 | -0.17 |
| BTC Daily | transformer | Transformer | 580 | 286 | 294 | 49.31% | 52.08% | 49.38% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 580 | 285 | 295 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 352 | 166 | 186 | 47.16% | 45.83% | 47.16% | 2.84 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 580 | 274 | 306 | 47.24% | 45.83% | 48.12% | 2.76 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 406 | 187 | 219 | 46.06% | 47.92% | 46.06% | 3.94 pp | -32 | 36 | -0.89 |
| BTC Market Hours | nn | NN | 352 | 160 | 192 | 45.45% | 47.50% | 45.45% | 4.55 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 406 | 186 | 220 | 45.81% | 45.83% | 45.81% | 4.19 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 406 | 185 | 221 | 45.57% | 47.08% | 45.57% | 4.43 pp | -36 | 36 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | rf | RandomForest | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Daily | lstm | LSTM | 580 | 260 | 320 | 44.83% | 45.42% | 44.58% | 5.17 pp | -60 | 37 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 352 | 146 | 206 | 41.48% | 42.08% | 41.48% | 8.52 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 406 | 167 | 239 | 41.13% | 39.58% | 41.13% | 8.87 pp | -72 | 36 | -2.00 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 580 | 250 | 330 | 43.10% | 44.17% | 43.75% | 6.90 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 37.92% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 3 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 590 | 237 | 353 | 40.17% | 35.00% | 40.62% | 9.83 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 580 | 286 | 294 | 49.31% | 52.08% | 49.38% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 580 | 285 | 295 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 580 | 274 | 306 | 47.24% | 45.83% | 48.12% | 2.76 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 580 | 260 | 320 | 44.83% | 45.42% | 44.58% | 5.17 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 580 | 250 | 330 | 43.10% | 44.17% | 43.75% | 6.90 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 590 | 237 | 353 | 40.17% | 35.00% | 40.62% | 9.83 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 352 | 173 | 179 | 49.15% | 47.08% | 49.15% | 0.85 pp | -6 | 36 | -0.17 |
| BTC Market Hours | transformer | Transformer | 352 | 166 | 186 | 47.16% | 45.83% | 47.16% | 2.84 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 352 | 160 | 192 | 45.45% | 47.50% | 45.45% | 4.55 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | rf | RandomForest | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 352 | 146 | 206 | 41.48% | 42.08% | 41.48% | 8.52 pp | -60 | 36 | -1.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 406 | 187 | 219 | 46.06% | 47.92% | 46.06% | 3.94 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 406 | 186 | 220 | 45.81% | 45.83% | 45.81% | 4.19 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 406 | 185 | 221 | 45.57% | 47.08% | 45.57% | 4.43 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 406 | 167 | 239 | 41.13% | 39.58% | 41.13% | 8.87 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 37.92% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | lstm | LSTM | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 3 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 3 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
