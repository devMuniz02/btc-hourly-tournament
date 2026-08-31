# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T01:06:38.237025+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1155 | 867 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1031 | 666 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 647 | 428 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 649 | 482 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 10:00:00+00:00 | 81 | 81 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 10:00:00+00:00 | 81 | 81 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 10:00:00+00:00 | 81 | 0 | 81 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 10:00:00+00:00 | 81 | 0 | 81 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 428 | 210 | 218 | 49.07% | 46.25% | 49.07% | 0.93 pp | -8 | 42 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 656 | 320 | 336 | 48.78% | 47.08% | 49.58% | 1.22 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 656 | 317 | 339 | 48.32% | 45.83% | 49.38% | 1.68 pp | -22 | 40 | -0.55 |
| BTC Market Hours | nn | NN | 428 | 202 | 226 | 47.20% | 50.42% | 47.20% | 2.80 pp | -24 | 42 | -0.57 |
| Consolidated Hourly | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 482 | 222 | 260 | 46.06% | 47.08% | 46.25% | 3.94 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 833 | 396 | 437 | 47.54% | 47.50% | 46.88% | 2.46 pp | -41 | 45 | -0.91 |
| BTC Market Hours | transformer | Transformer | 428 | 194 | 234 | 45.33% | 40.83% | 45.33% | 4.67 pp | -40 | 42 | -0.95 |
| BTC Market Hours Daily | nn | NN | 482 | 219 | 263 | 45.44% | 44.17% | 45.62% | 4.56 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 656 | 307 | 349 | 46.80% | 42.08% | 49.17% | 3.20 pp | -42 | 40 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 482 | 218 | 264 | 45.23% | 44.58% | 45.21% | 4.77 pp | -46 | 42 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 833 | 390 | 443 | 46.82% | 42.50% | 46.46% | 3.18 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 428 | 186 | 242 | 43.46% | 43.33% | 43.46% | 6.54 pp | -56 | 42 | -1.33 |
| BTC Market Hours | rf | RandomForest | 428 | 184 | 244 | 42.99% | 42.92% | 42.99% | 7.01 pp | -60 | 42 | -1.43 |
| BTC Hourly | nn | NN | 833 | 376 | 457 | 45.14% | 43.33% | 44.58% | 4.86 pp | -81 | 45 | -1.80 |
| BTC Daily | lstm | LSTM | 656 | 290 | 366 | 44.21% | 40.42% | 43.54% | 5.79 pp | -76 | 40 | -1.90 |
| BTC Hourly | rf | RandomForest | 833 | 372 | 461 | 44.66% | 42.92% | 44.17% | 5.34 pp | -89 | 45 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 482 | 199 | 283 | 41.29% | 41.67% | 41.25% | 8.71 pp | -84 | 42 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 428 | 170 | 258 | 39.72% | 37.50% | 39.72% | 10.28 pp | -88 | 42 | -2.10 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 482 | 195 | 287 | 40.46% | 39.17% | 40.42% | 9.54 pp | -92 | 42 | -2.19 |
| BTC Daily | rf | RandomForest | 656 | 281 | 375 | 42.84% | 41.25% | 43.96% | 7.16 pp | -94 | 40 | -2.35 |
| BTC Hourly | lstm | LSTM | 833 | 359 | 474 | 43.10% | 39.58% | 42.71% | 6.90 pp | -115 | 45 | -2.56 |
| BTC Market Hours Daily | xgb | XGBoost | 482 | 186 | 296 | 38.59% | 34.58% | 38.54% | 11.41 pp | -110 | 42 | -2.62 |
| BTC Hourly | xgb | XGBoost | 833 | 353 | 480 | 42.38% | 39.58% | 42.50% | 7.62 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 666 | 264 | 402 | 39.64% | 32.92% | 39.79% | 10.36 pp | -138 | 40 | -3.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 833 | 396 | 437 | 47.54% | 47.50% | 46.88% | 2.46 pp | -41 | 45 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 833 | 390 | 443 | 46.82% | 42.50% | 46.46% | 3.18 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 833 | 376 | 457 | 45.14% | 43.33% | 44.58% | 4.86 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 833 | 372 | 461 | 44.66% | 42.92% | 44.17% | 5.34 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 833 | 359 | 474 | 43.10% | 39.58% | 42.71% | 6.90 pp | -115 | 45 | -2.56 |
| BTC Hourly | xgb | XGBoost | 833 | 353 | 480 | 42.38% | 39.58% | 42.50% | 7.62 pp | -127 | 45 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 656 | 320 | 336 | 48.78% | 47.08% | 49.58% | 1.22 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 656 | 317 | 339 | 48.32% | 45.83% | 49.38% | 1.68 pp | -22 | 40 | -0.55 |
| BTC Daily | nn | NN | 656 | 307 | 349 | 46.80% | 42.08% | 49.17% | 3.20 pp | -42 | 40 | -1.05 |
| BTC Daily | lstm | LSTM | 656 | 290 | 366 | 44.21% | 40.42% | 43.54% | 5.79 pp | -76 | 40 | -1.90 |
| BTC Daily | rf | RandomForest | 656 | 281 | 375 | 42.84% | 41.25% | 43.96% | 7.16 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 666 | 264 | 402 | 39.64% | 32.92% | 39.79% | 10.36 pp | -138 | 40 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 428 | 210 | 218 | 49.07% | 46.25% | 49.07% | 0.93 pp | -8 | 42 | -0.19 |
| BTC Market Hours | nn | NN | 428 | 202 | 226 | 47.20% | 50.42% | 47.20% | 2.80 pp | -24 | 42 | -0.57 |
| BTC Market Hours | transformer | Transformer | 428 | 194 | 234 | 45.33% | 40.83% | 45.33% | 4.67 pp | -40 | 42 | -0.95 |
| BTC Market Hours | lstm | LSTM | 428 | 186 | 242 | 43.46% | 43.33% | 43.46% | 6.54 pp | -56 | 42 | -1.33 |
| BTC Market Hours | rf | RandomForest | 428 | 184 | 244 | 42.99% | 42.92% | 42.99% | 7.01 pp | -60 | 42 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 428 | 170 | 258 | 39.72% | 37.50% | 39.72% | 10.28 pp | -88 | 42 | -2.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 482 | 222 | 260 | 46.06% | 47.08% | 46.25% | 3.94 pp | -38 | 42 | -0.90 |
| BTC Market Hours Daily | nn | NN | 482 | 219 | 263 | 45.44% | 44.17% | 45.62% | 4.56 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 482 | 218 | 264 | 45.23% | 44.58% | 45.21% | 4.77 pp | -46 | 42 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 482 | 199 | 283 | 41.29% | 41.67% | 41.25% | 8.71 pp | -84 | 42 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 482 | 195 | 287 | 40.46% | 39.17% | 40.42% | 9.54 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 482 | 186 | 296 | 38.59% | 34.58% | 38.54% | 11.41 pp | -110 | 42 | -2.62 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 81 | 44 | 37 | 54.32% | 54.32% | 54.32% | 4.32 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 81 | 43 | 38 | 53.09% | 53.09% | 53.09% | 3.09 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 81 | 41 | 40 | 50.62% | 50.62% | 50.62% | 0.62 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 81 | 37 | 44 | 45.68% | 45.68% | 45.68% | 4.32 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | nn | NN | 81 | 32 | 49 | 39.51% | 39.51% | 39.51% | 10.49 pp | -17 | 8 | -2.12 |

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
