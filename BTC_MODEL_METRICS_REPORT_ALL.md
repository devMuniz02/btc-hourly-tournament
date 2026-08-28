# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T10:06:59.006860+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 818 | 301 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 981 | 616 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 558 | 378 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 560 | 432 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 378 | 186 | 192 | 49.21% | 47.92% | 49.21% | 0.79 pp | -6 | 38 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 606 | 297 | 309 | 49.01% | 48.33% | 50.21% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 606 | 297 | 309 | 49.01% | 50.00% | 49.79% | 0.99 pp | -12 | 38 | -0.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 378 | 177 | 201 | 46.83% | 44.58% | 46.83% | 3.17 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 378 | 175 | 203 | 46.30% | 49.17% | 46.30% | 3.70 pp | -28 | 38 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 784 | 372 | 412 | 47.45% | 45.42% | 47.92% | 2.55 pp | -40 | 43 | -0.93 |
| BTC Daily | nn | NN | 606 | 284 | 322 | 46.86% | 44.17% | 48.12% | 3.14 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 432 | 197 | 235 | 45.60% | 45.42% | 45.60% | 4.40 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 432 | 197 | 235 | 45.60% | 47.50% | 45.60% | 4.40 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 432 | 196 | 236 | 45.37% | 46.67% | 45.37% | 4.63 pp | -40 | 38 | -1.05 |
| BTC Hourly | transformer | Transformer | 784 | 368 | 416 | 46.94% | 43.33% | 46.04% | 3.06 pp | -48 | 43 | -1.12 |
| BTC Market Hours | lstm | LSTM | 378 | 162 | 216 | 42.86% | 43.75% | 42.86% | 7.14 pp | -54 | 38 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| BTC Market Hours | rf | RandomForest | 378 | 160 | 218 | 42.33% | 40.42% | 42.33% | 7.67 pp | -58 | 38 | -1.53 |
| BTC Daily | lstm | LSTM | 606 | 270 | 336 | 44.55% | 43.33% | 44.38% | 5.45 pp | -66 | 38 | -1.74 |
| BTC Hourly | nn | NN | 784 | 353 | 431 | 45.03% | 40.42% | 45.83% | 4.97 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 784 | 350 | 434 | 44.64% | 43.33% | 44.38% | 5.36 pp | -84 | 43 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 378 | 151 | 227 | 39.95% | 38.75% | 39.95% | 10.05 pp | -76 | 38 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| BTC Hourly | lstm | LSTM | 784 | 347 | 437 | 44.26% | 43.75% | 45.62% | 5.74 pp | -90 | 43 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 432 | 175 | 257 | 40.51% | 39.58% | 40.51% | 9.49 pp | -82 | 38 | -2.16 |
| BTC Daily | rf | RandomForest | 606 | 261 | 345 | 43.07% | 43.75% | 43.54% | 6.93 pp | -84 | 38 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 432 | 170 | 262 | 39.35% | 37.92% | 39.35% | 10.65 pp | -92 | 38 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 432 | 170 | 262 | 39.35% | 38.75% | 39.35% | 10.65 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 784 | 336 | 448 | 42.86% | 40.42% | 44.38% | 7.14 pp | -112 | 43 | -2.60 |
| BTC Daily | xgb | XGBoost | 616 | 246 | 370 | 39.94% | 34.17% | 40.21% | 10.06 pp | -124 | 38 | -3.26 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 784 | 372 | 412 | 47.45% | 45.42% | 47.92% | 2.55 pp | -40 | 43 | -0.93 |
| BTC Hourly | transformer | Transformer | 784 | 368 | 416 | 46.94% | 43.33% | 46.04% | 3.06 pp | -48 | 43 | -1.12 |
| BTC Hourly | nn | NN | 784 | 353 | 431 | 45.03% | 40.42% | 45.83% | 4.97 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 784 | 350 | 434 | 44.64% | 43.33% | 44.38% | 5.36 pp | -84 | 43 | -1.95 |
| BTC Hourly | lstm | LSTM | 784 | 347 | 437 | 44.26% | 43.75% | 45.62% | 5.74 pp | -90 | 43 | -2.09 |
| BTC Hourly | xgb | XGBoost | 784 | 336 | 448 | 42.86% | 40.42% | 44.38% | 7.14 pp | -112 | 43 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 606 | 297 | 309 | 49.01% | 48.33% | 50.21% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 606 | 297 | 309 | 49.01% | 50.00% | 49.79% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 606 | 284 | 322 | 46.86% | 44.17% | 48.12% | 3.14 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 606 | 270 | 336 | 44.55% | 43.33% | 44.38% | 5.45 pp | -66 | 38 | -1.74 |
| BTC Daily | rf | RandomForest | 606 | 261 | 345 | 43.07% | 43.75% | 43.54% | 6.93 pp | -84 | 38 | -2.21 |
| BTC Daily | xgb | XGBoost | 616 | 246 | 370 | 39.94% | 34.17% | 40.21% | 10.06 pp | -124 | 38 | -3.26 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 378 | 186 | 192 | 49.21% | 47.92% | 49.21% | 0.79 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 378 | 177 | 201 | 46.83% | 44.58% | 46.83% | 3.17 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 378 | 175 | 203 | 46.30% | 49.17% | 46.30% | 3.70 pp | -28 | 38 | -0.74 |
| BTC Market Hours | lstm | LSTM | 378 | 162 | 216 | 42.86% | 43.75% | 42.86% | 7.14 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 378 | 160 | 218 | 42.33% | 40.42% | 42.33% | 7.67 pp | -58 | 38 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 378 | 151 | 227 | 39.95% | 38.75% | 39.95% | 10.05 pp | -76 | 38 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 432 | 197 | 235 | 45.60% | 45.42% | 45.60% | 4.40 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 432 | 197 | 235 | 45.60% | 47.50% | 45.60% | 4.40 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 432 | 196 | 236 | 45.37% | 46.67% | 45.37% | 4.63 pp | -40 | 38 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 432 | 175 | 257 | 40.51% | 39.58% | 40.51% | 9.49 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 432 | 170 | 262 | 39.35% | 37.92% | 39.35% | 10.65 pp | -92 | 38 | -2.42 |
| BTC Market Hours Daily | xgb | XGBoost | 432 | 170 | 262 | 39.35% | 38.75% | 39.35% | 10.65 pp | -92 | 38 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

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
