# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T07:18:25.568045+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1016 | 651 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 619 | 413 | 205 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 413 | 205 | 208 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | mlp_sklearn | MLPClassifier | 641 | 311 | 330 | 48.52% | 45.42% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 641 | 311 | 330 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 413 | 195 | 218 | 47.22% | 50.42% | 47.22% | 2.78 pp | -23 | 41 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 467 | 217 | 250 | 46.47% | 46.67% | 46.47% | 3.53 pp | -33 | 41 | -0.80 |
| BTC Market Hours | transformer | Transformer | 413 | 190 | 223 | 46.00% | 41.67% | 46.00% | 4.00 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 467 | 214 | 253 | 45.82% | 45.83% | 45.82% | 4.18 pp | -39 | 41 | -0.95 |
| BTC Hourly | transformer | Transformer | 819 | 388 | 431 | 47.37% | 46.67% | 46.46% | 2.63 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 467 | 213 | 254 | 45.61% | 45.00% | 45.61% | 4.39 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| BTC Daily | nn | NN | 641 | 300 | 341 | 46.80% | 42.08% | 48.75% | 3.20 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 819 | 385 | 434 | 47.01% | 43.33% | 47.08% | 2.99 pp | -49 | 44 | -1.11 |
| BTC Market Hours | lstm | LSTM | 413 | 182 | 231 | 44.07% | 44.58% | 44.07% | 5.93 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 413 | 178 | 235 | 43.10% | 42.08% | 43.10% | 6.90 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 819 | 371 | 448 | 45.30% | 42.50% | 45.21% | 4.70 pp | -77 | 44 | -1.75 |
| BTC Daily | lstm | LSTM | 641 | 284 | 357 | 44.31% | 41.67% | 43.96% | 5.69 pp | -73 | 40 | -1.82 |
| BTC Hourly | rf | RandomForest | 819 | 366 | 453 | 44.69% | 44.58% | 44.38% | 5.31 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 467 | 192 | 275 | 41.11% | 41.67% | 41.11% | 8.89 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 467 | 191 | 276 | 40.90% | 40.00% | 40.90% | 9.10 pp | -85 | 41 | -2.07 |
| BTC Market Hours | xgb | XGBoost | 413 | 164 | 249 | 39.71% | 37.50% | 39.71% | 10.29 pp | -85 | 41 | -2.07 |
| BTC Daily | rf | RandomForest | 641 | 272 | 369 | 42.43% | 40.83% | 43.33% | 7.57 pp | -97 | 40 | -2.42 |
| BTC Hourly | lstm | LSTM | 819 | 356 | 463 | 43.47% | 41.67% | 43.96% | 6.53 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 467 | 181 | 286 | 38.76% | 35.42% | 38.76% | 11.24 pp | -105 | 41 | -2.56 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| BTC Hourly | xgb | XGBoost | 819 | 347 | 472 | 42.37% | 40.42% | 42.71% | 7.63 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 651 | 254 | 397 | 39.02% | 30.42% | 38.96% | 10.98 pp | -143 | 40 | -3.58 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 641 | 311 | 330 | 48.52% | 45.42% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 641 | 311 | 330 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 641 | 300 | 341 | 46.80% | 42.08% | 48.75% | 3.20 pp | -41 | 40 | -1.02 |
| BTC Daily | lstm | LSTM | 641 | 284 | 357 | 44.31% | 41.67% | 43.96% | 5.69 pp | -73 | 40 | -1.82 |
| BTC Daily | rf | RandomForest | 641 | 272 | 369 | 42.43% | 40.83% | 43.33% | 7.57 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 651 | 254 | 397 | 39.02% | 30.42% | 38.96% | 10.98 pp | -143 | 40 | -3.58 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 413 | 205 | 208 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 413 | 195 | 218 | 47.22% | 50.42% | 47.22% | 2.78 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 413 | 190 | 223 | 46.00% | 41.67% | 46.00% | 4.00 pp | -33 | 41 | -0.80 |
| BTC Market Hours | lstm | LSTM | 413 | 182 | 231 | 44.07% | 44.58% | 44.07% | 5.93 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 413 | 178 | 235 | 43.10% | 42.08% | 43.10% | 6.90 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 413 | 164 | 249 | 39.71% | 37.50% | 39.71% | 10.29 pp | -85 | 41 | -2.07 |

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
