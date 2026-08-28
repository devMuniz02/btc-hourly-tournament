# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T06:16:02.253345+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 815 | 304 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 979 | 614 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 556 | 376 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 557 | 429 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 604 | 298 | 306 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 376 | 184 | 192 | 48.94% | 47.50% | 48.94% | 1.06 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 604 | 296 | 308 | 49.01% | 48.33% | 50.21% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Market Hours | transformer | Transformer | 376 | 176 | 200 | 46.81% | 45.00% | 46.81% | 3.19 pp | -24 | 38 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 376 | 173 | 203 | 46.01% | 48.75% | 46.01% | 3.99 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 604 | 284 | 320 | 47.02% | 45.00% | 48.12% | 2.98 pp | -36 | 38 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 781 | 370 | 411 | 47.38% | 45.42% | 47.71% | 2.62 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 429 | 195 | 234 | 45.45% | 45.42% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 429 | 195 | 234 | 45.45% | 47.08% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 429 | 195 | 234 | 45.45% | 47.92% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 781 | 366 | 415 | 46.86% | 42.92% | 45.83% | 3.14 pp | -49 | 43 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 376 | 161 | 215 | 42.82% | 43.33% | 42.82% | 7.18 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 376 | 159 | 217 | 42.29% | 40.00% | 42.29% | 7.71 pp | -58 | 38 | -1.53 |
| BTC Daily | lstm | LSTM | 604 | 270 | 334 | 44.70% | 43.75% | 44.58% | 5.30 pp | -64 | 38 | -1.68 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| BTC Hourly | nn | NN | 781 | 351 | 430 | 44.94% | 40.83% | 45.83% | 5.06 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 781 | 350 | 431 | 44.81% | 44.17% | 44.38% | 5.19 pp | -81 | 43 | -1.88 |
| BTC Market Hours | xgb | XGBoost | 376 | 150 | 226 | 39.89% | 39.17% | 39.89% | 10.11 pp | -76 | 38 | -2.00 |
| BTC Hourly | lstm | LSTM | 781 | 346 | 435 | 44.30% | 44.17% | 45.62% | 5.70 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 429 | 174 | 255 | 40.56% | 40.42% | 40.56% | 9.44 pp | -81 | 38 | -2.13 |
| BTC Daily | rf | RandomForest | 604 | 261 | 343 | 43.21% | 44.58% | 43.75% | 6.79 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 781 | 336 | 445 | 43.02% | 40.83% | 44.38% | 6.98 pp | -109 | 43 | -2.53 |
| BTC Daily | xgb | XGBoost | 614 | 246 | 368 | 40.07% | 34.58% | 40.42% | 9.93 pp | -122 | 38 | -3.21 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 781 | 370 | 411 | 47.38% | 45.42% | 47.71% | 2.62 pp | -41 | 43 | -0.95 |
| BTC Hourly | transformer | Transformer | 781 | 366 | 415 | 46.86% | 42.92% | 45.83% | 3.14 pp | -49 | 43 | -1.14 |
| BTC Hourly | nn | NN | 781 | 351 | 430 | 44.94% | 40.83% | 45.83% | 5.06 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 781 | 350 | 431 | 44.81% | 44.17% | 44.38% | 5.19 pp | -81 | 43 | -1.88 |
| BTC Hourly | lstm | LSTM | 781 | 346 | 435 | 44.30% | 44.17% | 45.62% | 5.70 pp | -89 | 43 | -2.07 |
| BTC Hourly | xgb | XGBoost | 781 | 336 | 445 | 43.02% | 40.83% | 44.38% | 6.98 pp | -109 | 43 | -2.53 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 604 | 298 | 306 | 49.34% | 50.83% | 50.21% | 0.66 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 604 | 296 | 308 | 49.01% | 48.33% | 50.21% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 604 | 284 | 320 | 47.02% | 45.00% | 48.12% | 2.98 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 604 | 270 | 334 | 44.70% | 43.75% | 44.58% | 5.30 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 604 | 261 | 343 | 43.21% | 44.58% | 43.75% | 6.79 pp | -82 | 38 | -2.16 |
| BTC Daily | xgb | XGBoost | 614 | 246 | 368 | 40.07% | 34.58% | 40.42% | 9.93 pp | -122 | 38 | -3.21 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 376 | 184 | 192 | 48.94% | 47.50% | 48.94% | 1.06 pp | -8 | 38 | -0.21 |
| BTC Market Hours | transformer | Transformer | 376 | 176 | 200 | 46.81% | 45.00% | 46.81% | 3.19 pp | -24 | 38 | -0.63 |
| BTC Market Hours | nn | NN | 376 | 173 | 203 | 46.01% | 48.75% | 46.01% | 3.99 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 376 | 161 | 215 | 42.82% | 43.33% | 42.82% | 7.18 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 376 | 159 | 217 | 42.29% | 40.00% | 42.29% | 7.71 pp | -58 | 38 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 376 | 150 | 226 | 39.89% | 39.17% | 39.89% | 10.11 pp | -76 | 38 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 429 | 195 | 234 | 45.45% | 45.42% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 429 | 195 | 234 | 45.45% | 47.08% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 429 | 195 | 234 | 45.45% | 47.92% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 429 | 174 | 255 | 40.56% | 40.42% | 40.56% | 9.44 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

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
