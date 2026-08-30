# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T09:04:18.782910+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1018 | 653 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 621 | 415 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 623 | 469 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T11:00:00+00:00 | 71 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T11:00:00+00:00 | 71 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T11:00:00+00:00 | 71 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-24T11:00:00+00:00 | 72 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 415 | 206 | 209 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | mlp_sklearn | MLPClassifier | 643 | 312 | 331 | 48.52% | 45.83% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 643 | 312 | 331 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 415 | 196 | 219 | 47.23% | 50.83% | 47.23% | 2.77 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 415 | 191 | 224 | 46.02% | 42.08% | 46.02% | 3.98 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 469 | 217 | 252 | 46.27% | 46.25% | 46.27% | 3.73 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 469 | 215 | 254 | 45.84% | 45.83% | 45.84% | 4.16 pp | -39 | 41 | -0.95 |
| BTC Daily | nn | NN | 643 | 302 | 341 | 46.97% | 42.92% | 49.17% | 3.03 pp | -39 | 40 | -0.97 |
| BTC Hourly | transformer | Transformer | 820 | 388 | 432 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | nn | NN | 469 | 214 | 255 | 45.63% | 44.58% | 45.63% | 4.37 pp | -41 | 41 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 820 | 386 | 434 | 47.07% | 43.33% | 47.08% | 2.93 pp | -48 | 44 | -1.09 |
| BTC Market Hours | lstm | LSTM | 415 | 183 | 232 | 44.10% | 44.17% | 44.10% | 5.90 pp | -49 | 41 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 415 | 180 | 235 | 43.37% | 42.50% | 43.37% | 6.63 pp | -55 | 41 | -1.34 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 820 | 371 | 449 | 45.24% | 42.50% | 45.00% | 4.76 pp | -78 | 44 | -1.77 |
| BTC Daily | lstm | LSTM | 643 | 284 | 359 | 44.17% | 41.25% | 43.54% | 5.83 pp | -75 | 40 | -1.88 |
| BTC Hourly | rf | RandomForest | 820 | 367 | 453 | 44.76% | 45.00% | 44.58% | 5.24 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 469 | 194 | 275 | 41.36% | 42.08% | 41.36% | 8.64 pp | -81 | 41 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 415 | 165 | 250 | 39.76% | 37.08% | 39.76% | 10.24 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 469 | 191 | 278 | 40.72% | 39.58% | 40.72% | 9.28 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 643 | 274 | 369 | 42.61% | 41.25% | 43.54% | 7.39 pp | -95 | 40 | -2.38 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Hourly | lstm | LSTM | 820 | 356 | 464 | 43.41% | 41.25% | 43.75% | 6.59 pp | -108 | 44 | -2.45 |
| BTC Market Hours Daily | xgb | XGBoost | 469 | 182 | 287 | 38.81% | 35.42% | 38.81% | 11.19 pp | -105 | 41 | -2.56 |
| BTC Hourly | xgb | XGBoost | 820 | 347 | 473 | 42.32% | 40.00% | 42.71% | 7.68 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 653 | 256 | 397 | 39.20% | 30.83% | 39.17% | 10.80 pp | -141 | 40 | -3.52 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 643 | 312 | 331 | 48.52% | 45.83% | 50.00% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 643 | 312 | 331 | 48.52% | 45.83% | 49.17% | 1.48 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 643 | 302 | 341 | 46.97% | 42.92% | 49.17% | 3.03 pp | -39 | 40 | -0.97 |
| BTC Daily | lstm | LSTM | 643 | 284 | 359 | 44.17% | 41.25% | 43.54% | 5.83 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 643 | 274 | 369 | 42.61% | 41.25% | 43.54% | 7.39 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 653 | 256 | 397 | 39.20% | 30.83% | 39.17% | 10.80 pp | -141 | 40 | -3.52 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 415 | 206 | 209 | 49.64% | 48.33% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 415 | 196 | 219 | 47.23% | 50.83% | 47.23% | 2.77 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 415 | 191 | 224 | 46.02% | 42.08% | 46.02% | 3.98 pp | -33 | 41 | -0.80 |
| BTC Market Hours | lstm | LSTM | 415 | 183 | 232 | 44.10% | 44.17% | 44.10% | 5.90 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 415 | 180 | 235 | 43.37% | 42.50% | 43.37% | 6.63 pp | -55 | 41 | -1.34 |
| BTC Market Hours | xgb | XGBoost | 415 | 165 | 250 | 39.76% | 37.08% | 39.76% | 10.24 pp | -85 | 41 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 469 | 217 | 252 | 46.27% | 46.25% | 46.27% | 3.73 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 469 | 215 | 254 | 45.84% | 45.83% | 45.84% | 4.16 pp | -39 | 41 | -0.95 |
| BTC Market Hours Daily | nn | NN | 469 | 214 | 255 | 45.63% | 44.58% | 45.63% | 4.37 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 469 | 194 | 275 | 41.36% | 42.08% | 41.36% | 8.64 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 469 | 191 | 278 | 40.72% | 39.58% | 40.72% | 9.28 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 469 | 182 | 287 | 38.81% | 35.42% | 38.81% | 11.19 pp | -105 | 41 | -2.56 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
