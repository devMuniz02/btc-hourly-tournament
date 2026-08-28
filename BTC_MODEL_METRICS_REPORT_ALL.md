# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T02:32:56.711456+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 812 | 307 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 976 | 611 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 553 | 373 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 554 | 426 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 0 | 34 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 0 | 34 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| BTC Daily | transformer | Transformer | 601 | 297 | 304 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 601 | 283 | 318 | 47.09% | 45.00% | 48.33% | 2.91 pp | -35 | 38 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 778 | 368 | 410 | 47.30% | 44.58% | 47.50% | 2.70 pp | -42 | 42 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 426 | 194 | 232 | 45.54% | 45.42% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 426 | 194 | 232 | 45.54% | 46.67% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 426 | 194 | 232 | 45.54% | 47.92% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 778 | 365 | 413 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 601 | 268 | 333 | 44.59% | 42.92% | 44.79% | 5.41 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 778 | 349 | 429 | 44.86% | 40.42% | 45.62% | 5.14 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 778 | 348 | 430 | 44.73% | 43.33% | 44.38% | 5.27 pp | -82 | 42 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 601 | 261 | 340 | 43.43% | 45.00% | 44.17% | 6.57 pp | -79 | 38 | -2.08 |
| BTC Hourly | lstm | LSTM | 778 | 344 | 434 | 44.22% | 43.33% | 45.83% | 5.78 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | rf | RandomForest | 426 | 172 | 254 | 40.38% | 40.00% | 40.38% | 9.62 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 426 | 168 | 258 | 39.44% | 38.33% | 39.44% | 10.56 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 426 | 167 | 259 | 39.20% | 37.92% | 39.20% | 10.80 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 778 | 333 | 445 | 42.80% | 40.00% | 43.96% | 7.20 pp | -112 | 42 | -2.67 |
| BTC Daily | xgb | XGBoost | 611 | 246 | 365 | 40.26% | 35.83% | 40.42% | 9.74 pp | -119 | 38 | -3.13 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 778 | 368 | 410 | 47.30% | 44.58% | 47.50% | 2.70 pp | -42 | 42 | -1.00 |
| BTC Hourly | transformer | Transformer | 778 | 365 | 413 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 42 | -1.14 |
| BTC Hourly | nn | NN | 778 | 349 | 429 | 44.86% | 40.42% | 45.62% | 5.14 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 778 | 348 | 430 | 44.73% | 43.33% | 44.38% | 5.27 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 778 | 344 | 434 | 44.22% | 43.33% | 45.83% | 5.78 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 778 | 333 | 445 | 42.80% | 40.00% | 43.96% | 7.20 pp | -112 | 42 | -2.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 601 | 297 | 304 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 601 | 283 | 318 | 47.09% | 45.00% | 48.33% | 2.91 pp | -35 | 38 | -0.92 |
| BTC Daily | lstm | LSTM | 601 | 268 | 333 | 44.59% | 42.92% | 44.79% | 5.41 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 601 | 261 | 340 | 43.43% | 45.00% | 44.17% | 6.57 pp | -79 | 38 | -2.08 |
| BTC Daily | xgb | XGBoost | 611 | 246 | 365 | 40.26% | 35.83% | 40.42% | 9.74 pp | -119 | 38 | -3.13 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 426 | 194 | 232 | 45.54% | 45.42% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 426 | 194 | 232 | 45.54% | 46.67% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 426 | 194 | 232 | 45.54% | 47.92% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 426 | 172 | 254 | 40.38% | 40.00% | 40.38% | 9.62 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 426 | 168 | 258 | 39.44% | 38.33% | 39.44% | 10.56 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 426 | 167 | 259 | 39.20% | 37.92% | 39.20% | 10.80 pp | -92 | 38 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

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
