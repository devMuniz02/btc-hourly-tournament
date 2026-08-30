# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T12:31:12.921617+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1145 | 857 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1021 | 656 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 624 | 418 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 626 | 472 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 418 | 207 | 211 | 49.52% | 47.50% | 49.52% | 0.48 pp | -4 | 42 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 646 | 314 | 332 | 48.61% | 46.25% | 50.00% | 1.39 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 646 | 313 | 333 | 48.45% | 45.42% | 49.38% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 418 | 197 | 221 | 47.13% | 50.42% | 47.13% | 2.87 pp | -24 | 42 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 472 | 219 | 253 | 46.40% | 47.08% | 46.40% | 3.60 pp | -34 | 41 | -0.83 |
| BTC Market Hours | transformer | Transformer | 418 | 191 | 227 | 45.69% | 41.25% | 45.69% | 4.31 pp | -36 | 42 | -0.86 |
| BTC Hourly | transformer | Transformer | 823 | 390 | 433 | 47.39% | 46.67% | 46.46% | 2.61 pp | -43 | 44 | -0.98 |
| BTC Daily | nn | NN | 646 | 303 | 343 | 46.90% | 42.50% | 49.38% | 3.10 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | nn | NN | 472 | 215 | 257 | 45.55% | 44.58% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 472 | 215 | 257 | 45.55% | 45.00% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 823 | 386 | 437 | 46.90% | 42.50% | 46.67% | 3.10 pp | -51 | 44 | -1.16 |
| BTC Market Hours | lstm | LSTM | 418 | 183 | 235 | 43.78% | 43.75% | 43.78% | 6.22 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 418 | 181 | 237 | 43.30% | 42.92% | 43.30% | 6.70 pp | -56 | 42 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 823 | 372 | 451 | 45.20% | 42.50% | 44.79% | 4.80 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 646 | 286 | 360 | 44.27% | 42.08% | 43.75% | 5.73 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 472 | 196 | 276 | 41.53% | 42.50% | 41.53% | 8.47 pp | -80 | 41 | -1.95 |
| BTC Hourly | rf | RandomForest | 823 | 368 | 455 | 44.71% | 44.58% | 44.58% | 5.29 pp | -87 | 44 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 418 | 167 | 251 | 39.95% | 37.92% | 39.95% | 10.05 pp | -84 | 42 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 472 | 192 | 280 | 40.68% | 38.75% | 40.68% | 9.32 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 646 | 275 | 371 | 42.57% | 41.25% | 43.33% | 7.43 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Hourly | lstm | LSTM | 823 | 358 | 465 | 43.50% | 41.67% | 43.96% | 6.50 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 472 | 184 | 288 | 38.98% | 35.83% | 38.98% | 11.02 pp | -104 | 41 | -2.54 |
| BTC Hourly | xgb | XGBoost | 823 | 347 | 476 | 42.16% | 39.17% | 42.50% | 7.84 pp | -129 | 44 | -2.93 |
| BTC Daily | xgb | XGBoost | 656 | 258 | 398 | 39.33% | 31.67% | 39.38% | 10.67 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 823 | 390 | 433 | 47.39% | 46.67% | 46.46% | 2.61 pp | -43 | 44 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 823 | 386 | 437 | 46.90% | 42.50% | 46.67% | 3.10 pp | -51 | 44 | -1.16 |
| BTC Hourly | nn | NN | 823 | 372 | 451 | 45.20% | 42.50% | 44.79% | 4.80 pp | -79 | 44 | -1.80 |
| BTC Hourly | rf | RandomForest | 823 | 368 | 455 | 44.71% | 44.58% | 44.58% | 5.29 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 823 | 358 | 465 | 43.50% | 41.67% | 43.96% | 6.50 pp | -107 | 44 | -2.43 |
| BTC Hourly | xgb | XGBoost | 823 | 347 | 476 | 42.16% | 39.17% | 42.50% | 7.84 pp | -129 | 44 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 646 | 314 | 332 | 48.61% | 46.25% | 50.00% | 1.39 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 646 | 313 | 333 | 48.45% | 45.42% | 49.38% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 646 | 303 | 343 | 46.90% | 42.50% | 49.38% | 3.10 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 646 | 286 | 360 | 44.27% | 42.08% | 43.75% | 5.73 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 646 | 275 | 371 | 42.57% | 41.25% | 43.33% | 7.43 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 656 | 258 | 398 | 39.33% | 31.67% | 39.38% | 10.67 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 418 | 207 | 211 | 49.52% | 47.50% | 49.52% | 0.48 pp | -4 | 42 | -0.10 |
| BTC Market Hours | nn | NN | 418 | 197 | 221 | 47.13% | 50.42% | 47.13% | 2.87 pp | -24 | 42 | -0.57 |
| BTC Market Hours | transformer | Transformer | 418 | 191 | 227 | 45.69% | 41.25% | 45.69% | 4.31 pp | -36 | 42 | -0.86 |
| BTC Market Hours | lstm | LSTM | 418 | 183 | 235 | 43.78% | 43.75% | 43.78% | 6.22 pp | -52 | 42 | -1.24 |
| BTC Market Hours | rf | RandomForest | 418 | 181 | 237 | 43.30% | 42.92% | 43.30% | 6.70 pp | -56 | 42 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 418 | 167 | 251 | 39.95% | 37.92% | 39.95% | 10.05 pp | -84 | 42 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 472 | 219 | 253 | 46.40% | 47.08% | 46.40% | 3.60 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | nn | NN | 472 | 215 | 257 | 45.55% | 44.58% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 472 | 215 | 257 | 45.55% | 45.00% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 472 | 196 | 276 | 41.53% | 42.50% | 41.53% | 8.47 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 472 | 192 | 280 | 40.68% | 38.75% | 40.68% | 9.32 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 472 | 184 | 288 | 38.98% | 35.83% | 38.98% | 11.02 pp | -104 | 41 | -2.54 |

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
