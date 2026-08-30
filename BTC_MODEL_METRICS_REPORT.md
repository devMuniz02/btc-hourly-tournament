# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T03:40:31.131751+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1138 | 850 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1014 | 649 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 617 | 411 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 619 | 465 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 411 | 204 | 207 | 49.64% | 48.75% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 639 | 312 | 327 | 48.83% | 46.67% | 49.79% | 1.17 pp | -15 | 39 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 639 | 310 | 329 | 48.51% | 45.42% | 49.79% | 1.49 pp | -19 | 39 | -0.49 |
| BTC Market Hours | nn | NN | 411 | 194 | 217 | 47.20% | 50.42% | 47.20% | 2.80 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 411 | 190 | 221 | 46.23% | 42.08% | 46.23% | 3.77 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 465 | 216 | 249 | 46.45% | 46.67% | 46.45% | 3.55 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 465 | 214 | 251 | 46.02% | 46.25% | 46.02% | 3.98 pp | -37 | 41 | -0.90 |
| BTC Daily | nn | NN | 639 | 301 | 338 | 47.10% | 42.92% | 48.96% | 2.90 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | nn | NN | 465 | 212 | 253 | 45.59% | 45.42% | 45.59% | 4.41 pp | -41 | 41 | -1.00 |
| BTC Hourly | transformer | Transformer | 816 | 385 | 431 | 47.18% | 45.83% | 46.25% | 2.82 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 816 | 383 | 433 | 46.94% | 43.33% | 47.08% | 3.06 pp | -50 | 44 | -1.14 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 411 | 182 | 229 | 44.28% | 45.42% | 44.28% | 5.72 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 411 | 177 | 234 | 43.07% | 42.08% | 43.07% | 6.93 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 816 | 368 | 448 | 45.10% | 41.25% | 45.00% | 4.90 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 639 | 283 | 356 | 44.29% | 42.08% | 43.75% | 5.71 pp | -73 | 39 | -1.87 |
| BTC Hourly | rf | RandomForest | 816 | 365 | 451 | 44.73% | 44.58% | 44.58% | 5.27 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 465 | 191 | 274 | 41.08% | 41.67% | 41.08% | 8.92 pp | -83 | 41 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 465 | 190 | 275 | 40.86% | 40.00% | 40.86% | 9.14 pp | -85 | 41 | -2.07 |
| BTC Hourly | lstm | LSTM | 816 | 356 | 460 | 43.63% | 42.08% | 44.17% | 6.37 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 639 | 272 | 367 | 42.57% | 41.25% | 43.33% | 7.43 pp | -95 | 39 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 465 | 181 | 284 | 38.92% | 35.83% | 38.92% | 11.08 pp | -103 | 41 | -2.51 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| BTC Hourly | xgb | XGBoost | 816 | 345 | 471 | 42.28% | 39.58% | 42.71% | 7.72 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 649 | 253 | 396 | 38.98% | 30.42% | 38.96% | 11.02 pp | -143 | 39 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 816 | 385 | 431 | 47.18% | 45.83% | 46.25% | 2.82 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 816 | 383 | 433 | 46.94% | 43.33% | 47.08% | 3.06 pp | -50 | 44 | -1.14 |
| BTC Hourly | nn | NN | 816 | 368 | 448 | 45.10% | 41.25% | 45.00% | 4.90 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 816 | 365 | 451 | 44.73% | 44.58% | 44.58% | 5.27 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 816 | 356 | 460 | 43.63% | 42.08% | 44.17% | 6.37 pp | -104 | 44 | -2.36 |
| BTC Hourly | xgb | XGBoost | 816 | 345 | 471 | 42.28% | 39.58% | 42.71% | 7.72 pp | -126 | 44 | -2.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 639 | 312 | 327 | 48.83% | 46.67% | 49.79% | 1.17 pp | -15 | 39 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 639 | 310 | 329 | 48.51% | 45.42% | 49.79% | 1.49 pp | -19 | 39 | -0.49 |
| BTC Daily | nn | NN | 639 | 301 | 338 | 47.10% | 42.92% | 48.96% | 2.90 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 639 | 283 | 356 | 44.29% | 42.08% | 43.75% | 5.71 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 639 | 272 | 367 | 42.57% | 41.25% | 43.33% | 7.43 pp | -95 | 39 | -2.44 |
| BTC Daily | xgb | XGBoost | 649 | 253 | 396 | 38.98% | 30.42% | 38.96% | 11.02 pp | -143 | 39 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 411 | 204 | 207 | 49.64% | 48.75% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 411 | 194 | 217 | 47.20% | 50.42% | 47.20% | 2.80 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 411 | 190 | 221 | 46.23% | 42.08% | 46.23% | 3.77 pp | -31 | 41 | -0.76 |
| BTC Market Hours | lstm | LSTM | 411 | 182 | 229 | 44.28% | 45.42% | 44.28% | 5.72 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 411 | 177 | 234 | 43.07% | 42.08% | 43.07% | 6.93 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 41 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 465 | 216 | 249 | 46.45% | 46.67% | 46.45% | 3.55 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 465 | 214 | 251 | 46.02% | 46.25% | 46.02% | 3.98 pp | -37 | 41 | -0.90 |
| BTC Market Hours Daily | nn | NN | 465 | 212 | 253 | 45.59% | 45.42% | 45.59% | 4.41 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 465 | 191 | 274 | 41.08% | 41.67% | 41.08% | 8.92 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 465 | 190 | 275 | 40.86% | 40.00% | 40.86% | 9.14 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 465 | 181 | 284 | 38.92% | 35.83% | 38.92% | 11.08 pp | -103 | 41 | -2.51 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

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
