# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T02:24:02.885044+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 975 | 610 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 552 | 372 | 179 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 372 | 183 | 189 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Daily | transformer | Transformer | 600 | 296 | 304 | 49.33% | 50.83% | 50.00% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 600 | 293 | 307 | 48.83% | 47.50% | 49.58% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Market Hours | transformer | Transformer | 372 | 175 | 197 | 47.04% | 45.00% | 47.04% | 2.96 pp | -22 | 38 | -0.58 |
| BTC Market Hours | nn | NN | 372 | 171 | 201 | 45.97% | 48.75% | 45.97% | 4.03 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 600 | 282 | 318 | 47.00% | 44.58% | 48.33% | 3.00 pp | -36 | 38 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 778 | 368 | 410 | 47.30% | 44.58% | 47.50% | 2.70 pp | -42 | 42 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 426 | 194 | 232 | 45.54% | 45.42% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 426 | 194 | 232 | 45.54% | 46.67% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 426 | 194 | 232 | 45.54% | 47.92% | 45.54% | 4.46 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 778 | 365 | 413 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 372 | 159 | 213 | 42.74% | 42.92% | 42.74% | 7.26 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 372 | 158 | 214 | 42.47% | 40.83% | 42.47% | 7.53 pp | -56 | 38 | -1.47 |
| BTC Daily | lstm | LSTM | 600 | 268 | 332 | 44.67% | 43.33% | 45.00% | 5.33 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 778 | 349 | 429 | 44.86% | 40.42% | 45.62% | 5.14 pp | -80 | 42 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 372 | 149 | 223 | 40.05% | 40.00% | 40.05% | 9.95 pp | -74 | 38 | -1.95 |
| BTC Hourly | rf | RandomForest | 778 | 348 | 430 | 44.73% | 43.33% | 44.38% | 5.27 pp | -82 | 42 | -1.95 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 600 | 260 | 340 | 43.33% | 45.00% | 44.17% | 6.67 pp | -80 | 38 | -2.11 |
| BTC Hourly | lstm | LSTM | 778 | 344 | 434 | 44.22% | 43.33% | 45.83% | 5.78 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | rf | RandomForest | 426 | 172 | 254 | 40.38% | 40.00% | 40.38% | 9.62 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 426 | 168 | 258 | 39.44% | 38.33% | 39.44% | 10.56 pp | -90 | 38 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 426 | 167 | 259 | 39.20% | 37.92% | 39.20% | 10.80 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 778 | 333 | 445 | 42.80% | 40.00% | 43.96% | 7.20 pp | -112 | 42 | -2.67 |
| BTC Daily | xgb | XGBoost | 610 | 245 | 365 | 40.16% | 35.83% | 40.21% | 9.84 pp | -120 | 38 | -3.16 |
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
| BTC Daily | transformer | Transformer | 600 | 296 | 304 | 49.33% | 50.83% | 50.00% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 600 | 293 | 307 | 48.83% | 47.50% | 49.58% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 600 | 282 | 318 | 47.00% | 44.58% | 48.33% | 3.00 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 600 | 268 | 332 | 44.67% | 43.33% | 45.00% | 5.33 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 600 | 260 | 340 | 43.33% | 45.00% | 44.17% | 6.67 pp | -80 | 38 | -2.11 |
| BTC Daily | xgb | XGBoost | 610 | 245 | 365 | 40.16% | 35.83% | 40.21% | 9.84 pp | -120 | 38 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 372 | 183 | 189 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 372 | 175 | 197 | 47.04% | 45.00% | 47.04% | 2.96 pp | -22 | 38 | -0.58 |
| BTC Market Hours | nn | NN | 372 | 171 | 201 | 45.97% | 48.75% | 45.97% | 4.03 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 372 | 159 | 213 | 42.74% | 42.92% | 42.74% | 7.26 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 372 | 158 | 214 | 42.47% | 40.83% | 42.47% | 7.53 pp | -56 | 38 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 372 | 149 | 223 | 40.05% | 40.00% | 40.05% | 9.95 pp | -74 | 38 | -1.95 |

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
