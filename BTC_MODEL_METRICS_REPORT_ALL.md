# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T07:11:07.551916+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 797 | 322 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 961 | 596 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 358 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 527 | 412 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 17:00:00+00:00 | 23 | 23 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 17:00:00+00:00 | 23 | 23 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 17:00:00+00:00 | 23 | 1 | 22 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 17:00:00+00:00 | 23 | 1 | 22 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 586 | 291 | 295 | 49.66% | 52.08% | 50.21% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 287 | 299 | 48.98% | 47.92% | 49.58% | 1.02 pp | -12 | 37 | -0.32 |
| Consolidated Hourly | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Daily | nn | NN | 586 | 277 | 309 | 47.27% | 45.83% | 48.33% | 2.73 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 412 | 189 | 223 | 45.87% | 46.25% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 412 | 189 | 223 | 45.87% | 46.67% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 412 | 189 | 223 | 45.87% | 47.92% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 412 | 169 | 243 | 41.02% | 40.00% | 41.02% | 8.98 pp | -74 | 37 | -2.00 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 412 | 166 | 246 | 40.29% | 38.75% | 40.29% | 9.71 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 412 | 165 | 247 | 40.05% | 38.33% | 40.05% | 9.95 pp | -82 | 37 | -2.22 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |
| Consolidated Hourly | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 596 | 241 | 355 | 40.44% | 36.25% | 41.04% | 9.56 pp | -114 | 37 | -3.08 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 763 | 359 | 404 | 47.05% | 43.75% | 47.71% | 2.95 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 763 | 356 | 407 | 46.66% | 43.75% | 45.42% | 3.34 pp | -51 | 42 | -1.21 |
| BTC Hourly | nn | NN | 763 | 341 | 422 | 44.69% | 40.83% | 45.42% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 763 | 341 | 422 | 44.69% | 45.00% | 44.38% | 5.31 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 763 | 337 | 426 | 44.17% | 43.75% | 45.42% | 5.83 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 763 | 327 | 436 | 42.86% | 42.08% | 44.17% | 7.14 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 586 | 291 | 295 | 49.66% | 52.08% | 50.21% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 287 | 299 | 48.98% | 47.92% | 49.58% | 1.02 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 586 | 277 | 309 | 47.27% | 45.83% | 48.33% | 2.73 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 596 | 241 | 355 | 40.44% | 36.25% | 41.04% | 9.56 pp | -114 | 37 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 412 | 189 | 223 | 45.87% | 46.25% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 412 | 189 | 223 | 45.87% | 46.67% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 412 | 189 | 223 | 45.87% | 47.92% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 412 | 169 | 243 | 41.02% | 40.00% | 41.02% | 8.98 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 412 | 166 | 246 | 40.29% | 38.75% | 40.29% | 9.71 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 412 | 165 | 247 | 40.05% | 38.33% | 40.05% | 9.95 pp | -82 | 37 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
