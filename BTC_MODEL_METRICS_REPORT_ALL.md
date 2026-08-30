# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T12:13:08.740739+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1020 | 655 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 623 | 417 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 625 | 471 | 152 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 417 | 207 | 210 | 49.64% | 47.92% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | mlp_sklearn | MLPClassifier | 645 | 313 | 332 | 48.53% | 45.83% | 49.79% | 1.47 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 645 | 312 | 333 | 48.37% | 45.42% | 49.17% | 1.63 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 417 | 197 | 220 | 47.24% | 50.42% | 47.24% | 2.76 pp | -23 | 41 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 471 | 218 | 253 | 46.28% | 46.67% | 46.28% | 3.72 pp | -35 | 41 | -0.85 |
| BTC Market Hours | transformer | Transformer | 417 | 191 | 226 | 45.80% | 41.67% | 45.80% | 4.20 pp | -35 | 41 | -0.85 |
| BTC Hourly | transformer | Transformer | 823 | 390 | 433 | 47.39% | 46.67% | 46.46% | 2.61 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 471 | 215 | 256 | 45.65% | 45.00% | 45.65% | 4.35 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 471 | 215 | 256 | 45.65% | 45.42% | 45.65% | 4.35 pp | -41 | 41 | -1.00 |
| BTC Daily | nn | NN | 645 | 302 | 343 | 46.82% | 42.50% | 49.17% | 3.18 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 823 | 386 | 437 | 46.90% | 42.50% | 46.67% | 3.10 pp | -51 | 44 | -1.16 |
| BTC Market Hours | lstm | LSTM | 417 | 183 | 234 | 43.88% | 43.75% | 43.88% | 6.12 pp | -51 | 41 | -1.24 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 417 | 181 | 236 | 43.41% | 42.92% | 43.41% | 6.59 pp | -55 | 41 | -1.34 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 823 | 372 | 451 | 45.20% | 42.50% | 44.79% | 4.80 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 645 | 286 | 359 | 44.34% | 42.08% | 43.96% | 5.66 pp | -73 | 40 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 471 | 195 | 276 | 41.40% | 42.50% | 41.40% | 8.60 pp | -81 | 41 | -1.98 |
| BTC Hourly | rf | RandomForest | 823 | 368 | 455 | 44.71% | 44.58% | 44.58% | 5.29 pp | -87 | 44 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 417 | 167 | 250 | 40.05% | 37.92% | 40.05% | 9.95 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 471 | 192 | 279 | 40.76% | 39.17% | 40.76% | 9.24 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 645 | 274 | 371 | 42.48% | 40.83% | 43.12% | 7.52 pp | -97 | 40 | -2.42 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Hourly | lstm | LSTM | 823 | 358 | 465 | 43.50% | 41.67% | 43.96% | 6.50 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 471 | 183 | 288 | 38.85% | 35.83% | 38.85% | 11.15 pp | -105 | 41 | -2.56 |
| BTC Hourly | xgb | XGBoost | 823 | 347 | 476 | 42.16% | 39.17% | 42.50% | 7.84 pp | -129 | 44 | -2.93 |
| BTC Daily | xgb | XGBoost | 655 | 257 | 398 | 39.24% | 31.25% | 39.38% | 10.76 pp | -141 | 40 | -3.52 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 645 | 313 | 332 | 48.53% | 45.83% | 49.79% | 1.47 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 645 | 312 | 333 | 48.37% | 45.42% | 49.17% | 1.63 pp | -21 | 40 | -0.53 |
| BTC Daily | nn | NN | 645 | 302 | 343 | 46.82% | 42.50% | 49.17% | 3.18 pp | -41 | 40 | -1.02 |
| BTC Daily | lstm | LSTM | 645 | 286 | 359 | 44.34% | 42.08% | 43.96% | 5.66 pp | -73 | 40 | -1.82 |
| BTC Daily | rf | RandomForest | 645 | 274 | 371 | 42.48% | 40.83% | 43.12% | 7.52 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 655 | 257 | 398 | 39.24% | 31.25% | 39.38% | 10.76 pp | -141 | 40 | -3.52 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 417 | 207 | 210 | 49.64% | 47.92% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 417 | 197 | 220 | 47.24% | 50.42% | 47.24% | 2.76 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 417 | 191 | 226 | 45.80% | 41.67% | 45.80% | 4.20 pp | -35 | 41 | -0.85 |
| BTC Market Hours | lstm | LSTM | 417 | 183 | 234 | 43.88% | 43.75% | 43.88% | 6.12 pp | -51 | 41 | -1.24 |
| BTC Market Hours | rf | RandomForest | 417 | 181 | 236 | 43.41% | 42.92% | 43.41% | 6.59 pp | -55 | 41 | -1.34 |
| BTC Market Hours | xgb | XGBoost | 417 | 167 | 250 | 40.05% | 37.92% | 40.05% | 9.95 pp | -83 | 41 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 471 | 218 | 253 | 46.28% | 46.67% | 46.28% | 3.72 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | nn | NN | 471 | 215 | 256 | 45.65% | 45.00% | 45.65% | 4.35 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 471 | 215 | 256 | 45.65% | 45.42% | 45.65% | 4.35 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 471 | 195 | 276 | 41.40% | 42.50% | 41.40% | 8.60 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 471 | 192 | 279 | 40.76% | 39.17% | 40.76% | 9.24 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 471 | 183 | 288 | 38.85% | 35.83% | 38.85% | 11.15 pp | -105 | 41 | -2.56 |

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
