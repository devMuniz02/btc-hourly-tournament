# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T04:38:38.877632+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1139 | 851 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1014 | 649 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 617 | 411 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 619 | 465 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 0 | 67 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 07:00:00+00:00 | 67 | 0 | 67 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 411 | 204 | 207 | 49.64% | 48.75% | 49.64% | 0.36 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 639 | 311 | 328 | 48.67% | 46.25% | 49.58% | 1.33 pp | -17 | 39 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 639 | 309 | 330 | 48.36% | 45.00% | 49.58% | 1.64 pp | -21 | 39 | -0.54 |
| BTC Market Hours | nn | NN | 411 | 194 | 217 | 47.20% | 50.42% | 47.20% | 2.80 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 411 | 190 | 221 | 46.23% | 42.08% | 46.23% | 3.77 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 465 | 216 | 249 | 46.45% | 46.67% | 46.45% | 3.55 pp | -33 | 41 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 465 | 214 | 251 | 46.02% | 46.25% | 46.02% | 3.98 pp | -37 | 41 | -0.90 |
| BTC Daily | nn | NN | 639 | 300 | 339 | 46.95% | 42.50% | 48.75% | 3.05 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 465 | 212 | 253 | 45.59% | 45.42% | 45.59% | 4.41 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| BTC Hourly | transformer | Transformer | 817 | 386 | 431 | 47.25% | 46.25% | 46.25% | 2.75 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 817 | 384 | 433 | 47.00% | 43.75% | 47.08% | 3.00 pp | -49 | 44 | -1.11 |
| BTC Market Hours | lstm | LSTM | 411 | 182 | 229 | 44.28% | 45.42% | 44.28% | 5.72 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 411 | 177 | 234 | 43.07% | 42.08% | 43.07% | 6.93 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 817 | 369 | 448 | 45.17% | 41.67% | 45.00% | 4.83 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 639 | 284 | 355 | 44.44% | 42.50% | 43.96% | 5.56 pp | -71 | 39 | -1.82 |
| BTC Hourly | rf | RandomForest | 817 | 366 | 451 | 44.80% | 45.00% | 44.58% | 5.20 pp | -85 | 44 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 465 | 191 | 274 | 41.08% | 41.67% | 41.08% | 8.92 pp | -83 | 41 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 411 | 164 | 247 | 39.90% | 37.92% | 39.90% | 10.10 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 465 | 190 | 275 | 40.86% | 40.00% | 40.86% | 9.14 pp | -85 | 41 | -2.07 |
| BTC Hourly | lstm | LSTM | 817 | 356 | 461 | 43.57% | 42.08% | 44.17% | 6.43 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 639 | 271 | 368 | 42.41% | 40.83% | 43.12% | 7.59 pp | -97 | 39 | -2.49 |
| BTC Market Hours Daily | xgb | XGBoost | 465 | 181 | 284 | 38.92% | 35.83% | 38.92% | 11.08 pp | -103 | 41 | -2.51 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| BTC Hourly | xgb | XGBoost | 817 | 346 | 471 | 42.35% | 40.00% | 42.71% | 7.65 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 649 | 252 | 397 | 38.83% | 30.00% | 38.75% | 11.17 pp | -145 | 39 | -3.72 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 817 | 386 | 431 | 47.25% | 46.25% | 46.25% | 2.75 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 817 | 384 | 433 | 47.00% | 43.75% | 47.08% | 3.00 pp | -49 | 44 | -1.11 |
| BTC Hourly | nn | NN | 817 | 369 | 448 | 45.17% | 41.67% | 45.00% | 4.83 pp | -79 | 44 | -1.80 |
| BTC Hourly | rf | RandomForest | 817 | 366 | 451 | 44.80% | 45.00% | 44.58% | 5.20 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 817 | 356 | 461 | 43.57% | 42.08% | 44.17% | 6.43 pp | -105 | 44 | -2.39 |
| BTC Hourly | xgb | XGBoost | 817 | 346 | 471 | 42.35% | 40.00% | 42.71% | 7.65 pp | -125 | 44 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 639 | 311 | 328 | 48.67% | 46.25% | 49.58% | 1.33 pp | -17 | 39 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 639 | 309 | 330 | 48.36% | 45.00% | 49.58% | 1.64 pp | -21 | 39 | -0.54 |
| BTC Daily | nn | NN | 639 | 300 | 339 | 46.95% | 42.50% | 48.75% | 3.05 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 639 | 284 | 355 | 44.44% | 42.50% | 43.96% | 5.56 pp | -71 | 39 | -1.82 |
| BTC Daily | rf | RandomForest | 639 | 271 | 368 | 42.41% | 40.83% | 43.12% | 7.59 pp | -97 | 39 | -2.49 |
| BTC Daily | xgb | XGBoost | 649 | 252 | 397 | 38.83% | 30.00% | 38.75% | 11.17 pp | -145 | 39 | -3.72 |

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
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 36 | 31 | 53.73% | 53.73% | 53.73% | 3.73 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 30 | 37 | 44.78% | 44.78% | 44.78% | 5.22 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

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
