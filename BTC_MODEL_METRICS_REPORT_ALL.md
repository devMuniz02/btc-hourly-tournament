# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T17:36:32.238823+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 792 | 327 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 956 | 591 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 16:00:00+00:00 | 507 | 353 | 153 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 16:00:00+00:00 | 509 | 407 | 100 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 12:00:00+00:00 | 18 | 18 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 12:00:00+00:00 | 18 | 18 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 12:00:00+00:00 | 18 | 1 | 17 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 12:00:00+00:00 | 18 | 1 | 17 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 18 | 10 | 8 | 55.56% | 55.56% | 55.56% | 5.56 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 18 | 10 | 8 | 55.56% | 55.56% | 55.56% | 5.56 pp | 2 | 3 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 353 | 174 | 179 | 49.29% | 47.50% | 49.29% | 0.71 pp | -5 | 37 | -0.14 |
| BTC Daily | transformer | Transformer | 581 | 287 | 294 | 49.40% | 52.08% | 49.58% | 0.60 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 581 | 286 | 295 | 49.23% | 49.17% | 49.58% | 0.77 pp | -9 | 37 | -0.24 |
| BTC Market Hours | transformer | Transformer | 353 | 167 | 186 | 47.31% | 46.25% | 47.31% | 2.69 pp | -19 | 37 | -0.51 |
| Consolidated Hourly | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| BTC Market Hours | nn | NN | 353 | 161 | 192 | 45.61% | 47.92% | 45.61% | 4.39 pp | -31 | 37 | -0.84 |
| BTC Daily | nn | NN | 581 | 274 | 307 | 47.16% | 45.42% | 47.92% | 2.84 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 407 | 187 | 220 | 45.95% | 47.50% | 45.95% | 4.05 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 407 | 186 | 221 | 45.70% | 45.83% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 407 | 186 | 221 | 45.70% | 47.50% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 758 | 355 | 403 | 46.83% | 42.92% | 47.29% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 758 | 355 | 403 | 46.83% | 44.17% | 45.62% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 353 | 151 | 202 | 42.78% | 42.50% | 42.78% | 7.22 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 353 | 151 | 202 | 42.78% | 42.50% | 42.78% | 7.22 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 581 | 261 | 320 | 44.92% | 45.83% | 44.79% | 5.08 pp | -59 | 37 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 353 | 146 | 207 | 41.36% | 41.67% | 41.36% | 8.64 pp | -61 | 37 | -1.65 |
| BTC Hourly | rf | RandomForest | 758 | 340 | 418 | 44.85% | 45.00% | 44.58% | 5.15 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 758 | 338 | 420 | 44.59% | 40.83% | 45.00% | 5.41 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 407 | 168 | 239 | 41.28% | 40.00% | 41.28% | 8.72 pp | -71 | 36 | -1.97 |
| BTC Daily | rf | RandomForest | 581 | 251 | 330 | 43.20% | 44.58% | 43.75% | 6.80 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 758 | 333 | 425 | 43.93% | 42.50% | 45.42% | 6.07 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 407 | 163 | 244 | 40.05% | 38.33% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 407 | 163 | 244 | 40.05% | 37.50% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| BTC Hourly | xgb | XGBoost | 758 | 326 | 432 | 43.01% | 42.08% | 44.38% | 6.99 pp | -106 | 42 | -2.52 |
| Consolidated Hourly | nn | NN | 18 | 5 | 13 | 27.78% | 27.78% | 27.78% | 22.22 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 18 | 5 | 13 | 27.78% | 27.78% | 27.78% | 22.22 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 591 | 237 | 354 | 40.10% | 35.00% | 40.42% | 9.90 pp | -117 | 37 | -3.16 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 758 | 355 | 403 | 46.83% | 42.92% | 47.29% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 758 | 355 | 403 | 46.83% | 44.17% | 45.62% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | rf | RandomForest | 758 | 340 | 418 | 44.85% | 45.00% | 44.58% | 5.15 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 758 | 338 | 420 | 44.59% | 40.83% | 45.00% | 5.41 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 758 | 333 | 425 | 43.93% | 42.50% | 45.42% | 6.07 pp | -92 | 42 | -2.19 |
| BTC Hourly | xgb | XGBoost | 758 | 326 | 432 | 43.01% | 42.08% | 44.38% | 6.99 pp | -106 | 42 | -2.52 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 581 | 287 | 294 | 49.40% | 52.08% | 49.58% | 0.60 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 581 | 286 | 295 | 49.23% | 49.17% | 49.58% | 0.77 pp | -9 | 37 | -0.24 |
| BTC Daily | nn | NN | 581 | 274 | 307 | 47.16% | 45.42% | 47.92% | 2.84 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 581 | 261 | 320 | 44.92% | 45.83% | 44.79% | 5.08 pp | -59 | 37 | -1.59 |
| BTC Daily | rf | RandomForest | 581 | 251 | 330 | 43.20% | 44.58% | 43.75% | 6.80 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 591 | 237 | 354 | 40.10% | 35.00% | 40.42% | 9.90 pp | -117 | 37 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 353 | 174 | 179 | 49.29% | 47.50% | 49.29% | 0.71 pp | -5 | 37 | -0.14 |
| BTC Market Hours | transformer | Transformer | 353 | 167 | 186 | 47.31% | 46.25% | 47.31% | 2.69 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 353 | 161 | 192 | 45.61% | 47.92% | 45.61% | 4.39 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 353 | 151 | 202 | 42.78% | 42.50% | 42.78% | 7.22 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 353 | 151 | 202 | 42.78% | 42.50% | 42.78% | 7.22 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 353 | 146 | 207 | 41.36% | 41.67% | 41.36% | 8.64 pp | -61 | 37 | -1.65 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 407 | 187 | 220 | 45.95% | 47.50% | 45.95% | 4.05 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 407 | 186 | 221 | 45.70% | 45.83% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 407 | 186 | 221 | 45.70% | 47.50% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 407 | 168 | 239 | 41.28% | 40.00% | 41.28% | 8.72 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | lstm | LSTM | 407 | 163 | 244 | 40.05% | 38.33% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 407 | 163 | 244 | 40.05% | 37.50% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 18 | 10 | 8 | 55.56% | 55.56% | 55.56% | 5.56 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 18 | 5 | 13 | 27.78% | 27.78% | 27.78% | 22.22 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 18 | 12 | 6 | 66.67% | 66.67% | 66.67% | 16.67 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 18 | 11 | 7 | 61.11% | 61.11% | 61.11% | 11.11 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 18 | 10 | 8 | 55.56% | 55.56% | 55.56% | 5.56 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 18 | 5 | 13 | 27.78% | 27.78% | 27.78% | 22.22 pp | -8 | 3 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
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
