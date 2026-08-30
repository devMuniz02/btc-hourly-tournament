# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T07:27:39.285509+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1141 | 853 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1017 | 652 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 620 | 414 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 621 | 467 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 414 | 205 | 209 | 49.52% | 48.33% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 642 | 312 | 330 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 642 | 312 | 330 | 48.60% | 45.83% | 49.38% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Market Hours | nn | NN | 414 | 196 | 218 | 47.34% | 50.83% | 47.34% | 2.66 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 414 | 191 | 223 | 46.14% | 42.08% | 46.14% | 3.86 pp | -32 | 41 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 467 | 217 | 250 | 46.47% | 46.67% | 46.47% | 3.53 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 467 | 214 | 253 | 45.82% | 45.83% | 45.82% | 4.18 pp | -39 | 41 | -0.95 |
| BTC Hourly | transformer | Transformer | 819 | 388 | 431 | 47.37% | 46.67% | 46.46% | 2.63 pp | -43 | 44 | -0.98 |
| BTC Daily | nn | NN | 642 | 301 | 341 | 46.88% | 42.50% | 48.96% | 3.12 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | nn | NN | 467 | 213 | 254 | 45.61% | 45.00% | 45.61% | 4.39 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 819 | 385 | 434 | 47.01% | 43.33% | 47.08% | 2.99 pp | -49 | 44 | -1.11 |
| BTC Market Hours | lstm | LSTM | 414 | 182 | 232 | 43.96% | 44.17% | 43.96% | 6.04 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 414 | 179 | 235 | 43.24% | 42.50% | 43.24% | 6.76 pp | -56 | 41 | -1.37 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 819 | 371 | 448 | 45.30% | 42.50% | 45.21% | 4.70 pp | -77 | 44 | -1.75 |
| BTC Daily | lstm | LSTM | 642 | 284 | 358 | 44.24% | 41.67% | 43.75% | 5.76 pp | -74 | 40 | -1.85 |
| BTC Hourly | rf | RandomForest | 819 | 366 | 453 | 44.69% | 44.58% | 44.38% | 5.31 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 467 | 192 | 275 | 41.11% | 41.67% | 41.11% | 8.89 pp | -83 | 41 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 414 | 165 | 249 | 39.86% | 37.50% | 39.86% | 10.14 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 467 | 191 | 276 | 40.90% | 40.00% | 40.90% | 9.10 pp | -85 | 41 | -2.07 |
| BTC Daily | rf | RandomForest | 642 | 273 | 369 | 42.52% | 40.83% | 43.33% | 7.48 pp | -96 | 40 | -2.40 |
| BTC Hourly | lstm | LSTM | 819 | 356 | 463 | 43.47% | 41.67% | 43.96% | 6.53 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 467 | 181 | 286 | 38.76% | 35.42% | 38.76% | 11.24 pp | -105 | 41 | -2.56 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| BTC Hourly | xgb | XGBoost | 819 | 347 | 472 | 42.37% | 40.42% | 42.71% | 7.63 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 652 | 255 | 397 | 39.11% | 30.42% | 38.96% | 10.89 pp | -142 | 40 | -3.55 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 819 | 388 | 431 | 47.37% | 46.67% | 46.46% | 2.63 pp | -43 | 44 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 819 | 385 | 434 | 47.01% | 43.33% | 47.08% | 2.99 pp | -49 | 44 | -1.11 |
| BTC Hourly | nn | NN | 819 | 371 | 448 | 45.30% | 42.50% | 45.21% | 4.70 pp | -77 | 44 | -1.75 |
| BTC Hourly | rf | RandomForest | 819 | 366 | 453 | 44.69% | 44.58% | 44.38% | 5.31 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 819 | 356 | 463 | 43.47% | 41.67% | 43.96% | 6.53 pp | -107 | 44 | -2.43 |
| BTC Hourly | xgb | XGBoost | 819 | 347 | 472 | 42.37% | 40.42% | 42.71% | 7.63 pp | -125 | 44 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 642 | 312 | 330 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 642 | 312 | 330 | 48.60% | 45.83% | 49.38% | 1.40 pp | -18 | 40 | -0.45 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 467 | 217 | 250 | 46.47% | 46.67% | 46.47% | 3.53 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 467 | 214 | 253 | 45.82% | 45.83% | 45.82% | 4.18 pp | -39 | 41 | -0.95 |
| BTC Market Hours Daily | nn | NN | 467 | 213 | 254 | 45.61% | 45.00% | 45.61% | 4.39 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 467 | 192 | 275 | 41.11% | 41.67% | 41.11% | 8.89 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 467 | 191 | 276 | 40.90% | 40.00% | 40.90% | 9.10 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 467 | 181 | 286 | 38.76% | 35.42% | 38.76% | 11.24 pp | -105 | 41 | -2.56 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

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
