# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T08:38:06.224751+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1142 | 854 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1017 | 652 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 620 | 414 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 622 | 468 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 414 | 205 | 209 | 49.52% | 48.33% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 642 | 311 | 331 | 48.44% | 45.42% | 49.79% | 1.56 pp | -20 | 40 | -0.50 |
| BTC Daily | transformer | Transformer | 642 | 311 | 331 | 48.44% | 45.42% | 49.17% | 1.56 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 414 | 196 | 218 | 47.34% | 50.83% | 47.34% | 2.66 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 414 | 191 | 223 | 46.14% | 42.08% | 46.14% | 3.86 pp | -32 | 41 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 468 | 217 | 251 | 46.37% | 46.67% | 46.37% | 3.63 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 468 | 214 | 254 | 45.73% | 45.42% | 45.73% | 4.27 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 820 | 388 | 432 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 642 | 301 | 341 | 46.88% | 42.50% | 48.96% | 3.12 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | nn | NN | 468 | 213 | 255 | 45.51% | 44.58% | 45.51% | 4.49 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 820 | 386 | 434 | 47.07% | 43.33% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 414 | 182 | 232 | 43.96% | 44.17% | 43.96% | 6.04 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 414 | 179 | 235 | 43.24% | 42.50% | 43.24% | 6.76 pp | -56 | 41 | -1.37 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 820 | 371 | 449 | 45.24% | 42.50% | 45.00% | 4.76 pp | -78 | 44 | -1.77 |
| BTC Daily | lstm | LSTM | 642 | 284 | 358 | 44.24% | 41.67% | 43.75% | 5.76 pp | -74 | 40 | -1.85 |
| BTC Hourly | rf | RandomForest | 820 | 367 | 453 | 44.76% | 45.00% | 44.58% | 5.24 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 468 | 193 | 275 | 41.24% | 42.08% | 41.24% | 8.76 pp | -82 | 41 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 414 | 165 | 249 | 39.86% | 37.50% | 39.86% | 10.14 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 468 | 191 | 277 | 40.81% | 40.00% | 40.81% | 9.19 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 642 | 273 | 369 | 42.52% | 40.83% | 43.33% | 7.48 pp | -96 | 40 | -2.40 |
| BTC Hourly | lstm | LSTM | 820 | 356 | 464 | 43.41% | 41.25% | 43.75% | 6.59 pp | -108 | 44 | -2.45 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | xgb | XGBoost | 468 | 181 | 287 | 38.68% | 35.42% | 38.68% | 11.32 pp | -106 | 41 | -2.59 |
| BTC Hourly | xgb | XGBoost | 820 | 347 | 473 | 42.32% | 40.00% | 42.71% | 7.68 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 652 | 255 | 397 | 39.11% | 30.42% | 38.96% | 10.89 pp | -142 | 40 | -3.55 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 820 | 388 | 432 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 820 | 386 | 434 | 47.07% | 43.33% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 820 | 371 | 449 | 45.24% | 42.50% | 45.00% | 4.76 pp | -78 | 44 | -1.77 |
| BTC Hourly | rf | RandomForest | 820 | 367 | 453 | 44.76% | 45.00% | 44.58% | 5.24 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 820 | 356 | 464 | 43.41% | 41.25% | 43.75% | 6.59 pp | -108 | 44 | -2.45 |
| BTC Hourly | xgb | XGBoost | 820 | 347 | 473 | 42.32% | 40.00% | 42.71% | 7.68 pp | -126 | 44 | -2.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 642 | 311 | 331 | 48.44% | 45.42% | 49.79% | 1.56 pp | -20 | 40 | -0.50 |
| BTC Daily | transformer | Transformer | 642 | 311 | 331 | 48.44% | 45.42% | 49.17% | 1.56 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 642 | 301 | 341 | 46.88% | 42.50% | 48.96% | 3.12 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 642 | 284 | 358 | 44.24% | 41.67% | 43.75% | 5.76 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 642 | 273 | 369 | 42.52% | 40.83% | 43.33% | 7.48 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 652 | 255 | 397 | 39.11% | 30.42% | 38.96% | 10.89 pp | -142 | 40 | -3.55 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 414 | 205 | 209 | 49.52% | 48.33% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Market Hours | nn | NN | 414 | 196 | 218 | 47.34% | 50.83% | 47.34% | 2.66 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 414 | 191 | 223 | 46.14% | 42.08% | 46.14% | 3.86 pp | -32 | 41 | -0.78 |
| BTC Market Hours | lstm | LSTM | 414 | 182 | 232 | 43.96% | 44.17% | 43.96% | 6.04 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 414 | 179 | 235 | 43.24% | 42.50% | 43.24% | 6.76 pp | -56 | 41 | -1.37 |
| BTC Market Hours | xgb | XGBoost | 414 | 165 | 249 | 39.86% | 37.50% | 39.86% | 10.14 pp | -84 | 41 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 468 | 217 | 251 | 46.37% | 46.67% | 46.37% | 3.63 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 468 | 214 | 254 | 45.73% | 45.42% | 45.73% | 4.27 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | nn | NN | 468 | 213 | 255 | 45.51% | 44.58% | 45.51% | 4.49 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 468 | 193 | 275 | 41.24% | 42.08% | 41.24% | 8.76 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 468 | 191 | 277 | 40.81% | 40.00% | 40.81% | 9.19 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 468 | 181 | 287 | 38.68% | 35.42% | 38.68% | 11.32 pp | -106 | 41 | -2.59 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

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
