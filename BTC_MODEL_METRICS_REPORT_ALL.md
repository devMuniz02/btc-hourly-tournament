# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T04:43:07.314254+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 813 | 306 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 977 | 612 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 554 | 374 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 556 | 428 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 1 | 36 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 20:00:00+00:00 | 37 | 1 | 36 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 602 | 297 | 305 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 374 | 183 | 191 | 48.93% | 47.92% | 48.93% | 1.07 pp | -8 | 38 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 602 | 295 | 307 | 49.00% | 47.92% | 50.00% | 1.00 pp | -12 | 38 | -0.32 |
| BTC Market Hours | transformer | Transformer | 374 | 175 | 199 | 46.79% | 44.58% | 46.79% | 3.21 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 374 | 173 | 201 | 46.26% | 48.75% | 46.26% | 3.74 pp | -28 | 38 | -0.74 |
| Consolidated Hourly | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 602 | 283 | 319 | 47.01% | 45.00% | 48.12% | 2.99 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 428 | 195 | 233 | 45.56% | 45.83% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 428 | 195 | 233 | 45.56% | 48.33% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 779 | 368 | 411 | 47.24% | 44.58% | 47.50% | 2.76 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | nn | NN | 428 | 194 | 234 | 45.33% | 46.67% | 45.33% | 4.67 pp | -40 | 38 | -1.05 |
| BTC Hourly | transformer | Transformer | 779 | 365 | 414 | 46.85% | 42.92% | 45.83% | 3.15 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 374 | 160 | 214 | 42.78% | 43.33% | 42.78% | 7.22 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 374 | 159 | 215 | 42.51% | 40.42% | 42.51% | 7.49 pp | -56 | 38 | -1.47 |
| BTC Daily | lstm | LSTM | 602 | 269 | 333 | 44.68% | 43.33% | 44.79% | 5.32 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 779 | 350 | 429 | 44.93% | 40.83% | 45.83% | 5.07 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 779 | 349 | 430 | 44.80% | 43.75% | 44.38% | 5.20 pp | -81 | 42 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 374 | 149 | 225 | 39.84% | 39.17% | 39.84% | 10.16 pp | -76 | 38 | -2.00 |
| BTC Daily | rf | RandomForest | 602 | 261 | 341 | 43.36% | 44.58% | 43.96% | 6.64 pp | -80 | 38 | -2.11 |
| BTC Hourly | lstm | LSTM | 779 | 345 | 434 | 44.29% | 43.75% | 45.83% | 5.71 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | rf | RandomForest | 428 | 173 | 255 | 40.42% | 40.42% | 40.42% | 9.58 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 779 | 334 | 445 | 42.88% | 40.42% | 43.96% | 7.12 pp | -111 | 42 | -2.64 |
| Consolidated Hourly | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |
| BTC Daily | xgb | XGBoost | 612 | 246 | 366 | 40.20% | 35.42% | 40.42% | 9.80 pp | -120 | 38 | -3.16 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 779 | 368 | 411 | 47.24% | 44.58% | 47.50% | 2.76 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 779 | 365 | 414 | 46.85% | 42.92% | 45.83% | 3.15 pp | -49 | 42 | -1.17 |
| BTC Hourly | nn | NN | 779 | 350 | 429 | 44.93% | 40.83% | 45.83% | 5.07 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 779 | 349 | 430 | 44.80% | 43.75% | 44.38% | 5.20 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 779 | 345 | 434 | 44.29% | 43.75% | 45.83% | 5.71 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 779 | 334 | 445 | 42.88% | 40.42% | 43.96% | 7.12 pp | -111 | 42 | -2.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 602 | 297 | 305 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 602 | 295 | 307 | 49.00% | 47.92% | 50.00% | 1.00 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 602 | 283 | 319 | 47.01% | 45.00% | 48.12% | 2.99 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 602 | 269 | 333 | 44.68% | 43.33% | 44.79% | 5.32 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 602 | 261 | 341 | 43.36% | 44.58% | 43.96% | 6.64 pp | -80 | 38 | -2.11 |
| BTC Daily | xgb | XGBoost | 612 | 246 | 366 | 40.20% | 35.42% | 40.42% | 9.80 pp | -120 | 38 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 374 | 183 | 191 | 48.93% | 47.92% | 48.93% | 1.07 pp | -8 | 38 | -0.21 |
| BTC Market Hours | transformer | Transformer | 374 | 175 | 199 | 46.79% | 44.58% | 46.79% | 3.21 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 374 | 173 | 201 | 46.26% | 48.75% | 46.26% | 3.74 pp | -28 | 38 | -0.74 |
| BTC Market Hours | lstm | LSTM | 374 | 160 | 214 | 42.78% | 43.33% | 42.78% | 7.22 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 374 | 159 | 215 | 42.51% | 40.42% | 42.51% | 7.49 pp | -56 | 38 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 374 | 149 | 225 | 39.84% | 39.17% | 39.84% | 10.16 pp | -76 | 38 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 428 | 195 | 233 | 45.56% | 45.83% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 428 | 195 | 233 | 45.56% | 48.33% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 428 | 194 | 234 | 45.33% | 46.67% | 45.33% | 4.67 pp | -40 | 38 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 428 | 173 | 255 | 40.42% | 40.42% | 40.42% | 9.58 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 13 | 24 | 35.14% | 35.14% | 35.14% | 14.86 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
