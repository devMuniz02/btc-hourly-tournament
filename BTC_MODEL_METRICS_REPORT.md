# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T05:01:38.176412+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 814 | 305 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 977 | 612 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 554 | 374 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 556 | 428 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 0 | 36 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 0 | 36 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| BTC Daily | transformer | Transformer | 602 | 297 | 305 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 374 | 183 | 191 | 48.93% | 47.92% | 48.93% | 1.07 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 602 | 294 | 308 | 48.84% | 47.50% | 49.79% | 1.16 pp | -14 | 38 | -0.37 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 374 | 175 | 199 | 46.79% | 44.58% | 46.79% | 3.21 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 374 | 173 | 201 | 46.26% | 48.75% | 46.26% | 3.74 pp | -28 | 38 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 369 | 411 | 47.31% | 45.00% | 47.50% | 2.69 pp | -42 | 42 | -1.00 |
| BTC Daily | nn | NN | 602 | 282 | 320 | 46.84% | 44.58% | 47.92% | 3.16 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 428 | 195 | 233 | 45.56% | 45.83% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 428 | 195 | 233 | 45.56% | 48.33% | 45.56% | 4.44 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | nn | NN | 428 | 194 | 234 | 45.33% | 46.67% | 45.33% | 4.67 pp | -40 | 38 | -1.05 |
| BTC Hourly | transformer | Transformer | 780 | 365 | 415 | 46.79% | 42.50% | 45.83% | 3.21 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 374 | 160 | 214 | 42.78% | 43.33% | 42.78% | 7.22 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 374 | 159 | 215 | 42.51% | 40.42% | 42.51% | 7.49 pp | -56 | 38 | -1.47 |
| Consolidated Hourly | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| BTC Daily | lstm | LSTM | 602 | 269 | 333 | 44.68% | 43.33% | 44.79% | 5.32 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 780 | 349 | 431 | 44.74% | 43.75% | 44.17% | 5.26 pp | -82 | 42 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 374 | 149 | 225 | 39.84% | 39.17% | 39.84% | 10.16 pp | -76 | 38 | -2.00 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 42 | -2.10 |
| BTC Daily | rf | RandomForest | 602 | 260 | 342 | 43.19% | 44.17% | 43.75% | 6.81 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | rf | RandomForest | 428 | 173 | 255 | 40.42% | 40.42% | 40.42% | 9.58 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 428 | 168 | 260 | 39.25% | 38.33% | 39.25% | 10.75 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 780 | 335 | 445 | 42.95% | 40.42% | 44.17% | 7.05 pp | -110 | 42 | -2.62 |
| Consolidated Hourly | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 612 | 245 | 367 | 40.03% | 35.00% | 40.21% | 9.97 pp | -122 | 38 | -3.21 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 369 | 411 | 47.31% | 45.00% | 47.50% | 2.69 pp | -42 | 42 | -1.00 |
| BTC Hourly | transformer | Transformer | 780 | 365 | 415 | 46.79% | 42.50% | 45.83% | 3.21 pp | -50 | 42 | -1.19 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 780 | 349 | 431 | 44.74% | 43.75% | 44.17% | 5.26 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 42 | -2.10 |
| BTC Hourly | xgb | XGBoost | 780 | 335 | 445 | 42.95% | 40.42% | 44.17% | 7.05 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 602 | 297 | 305 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 602 | 294 | 308 | 48.84% | 47.50% | 49.79% | 1.16 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 602 | 282 | 320 | 46.84% | 44.58% | 47.92% | 3.16 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 602 | 269 | 333 | 44.68% | 43.33% | 44.79% | 5.32 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 602 | 260 | 342 | 43.19% | 44.17% | 43.75% | 6.81 pp | -82 | 38 | -2.16 |
| BTC Daily | xgb | XGBoost | 612 | 245 | 367 | 40.03% | 35.00% | 40.21% | 9.97 pp | -122 | 38 | -3.21 |

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
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
