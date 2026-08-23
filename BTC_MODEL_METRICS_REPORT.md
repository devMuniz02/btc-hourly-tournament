# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T18:31:56.188915+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 793 | 326 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 957 | 592 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 17:00:00+00:00 | 509 | 354 | 154 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 17:00:00+00:00 | 510 | 407 | 101 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 01:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 01:00:00+00:00 | 17 | 17 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 01:00:00+00:00 | 17 | 0 | 17 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 01:00:00+00:00 | 17 | 0 | 17 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 354 | 175 | 179 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | transformer | Transformer | 582 | 288 | 294 | 49.48% | 52.08% | 49.79% | 0.52 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 582 | 287 | 295 | 49.31% | 49.17% | 49.79% | 0.69 pp | -8 | 37 | -0.22 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 354 | 167 | 187 | 47.18% | 46.25% | 47.18% | 2.82 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 354 | 162 | 192 | 45.76% | 47.92% | 45.76% | 4.24 pp | -30 | 37 | -0.81 |
| BTC Daily | nn | NN | 582 | 275 | 307 | 47.25% | 45.42% | 48.12% | 2.75 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 407 | 187 | 220 | 45.95% | 47.50% | 45.95% | 4.05 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 407 | 186 | 221 | 45.70% | 45.83% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 407 | 186 | 221 | 45.70% | 47.50% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 3 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 759 | 356 | 403 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 759 | 355 | 404 | 46.77% | 44.17% | 45.42% | 3.23 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Daily | lstm | LSTM | 582 | 262 | 320 | 45.02% | 45.83% | 45.00% | 4.98 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 354 | 147 | 207 | 41.53% | 42.08% | 41.53% | 8.47 pp | -60 | 37 | -1.62 |
| BTC Hourly | rf | RandomForest | 759 | 340 | 419 | 44.80% | 44.58% | 44.38% | 5.20 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 759 | 339 | 420 | 44.66% | 41.25% | 45.21% | 5.34 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 407 | 168 | 239 | 41.28% | 40.00% | 41.28% | 8.72 pp | -71 | 36 | -1.97 |
| BTC Daily | rf | RandomForest | 582 | 252 | 330 | 43.30% | 44.58% | 43.96% | 6.70 pp | -78 | 37 | -2.11 |
| BTC Hourly | lstm | LSTM | 759 | 334 | 425 | 44.01% | 42.92% | 45.42% | 5.99 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 407 | 163 | 244 | 40.05% | 38.33% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 407 | 163 | 244 | 40.05% | 37.50% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 3 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 3 | -2.33 |
| BTC Hourly | xgb | XGBoost | 759 | 326 | 433 | 42.95% | 41.67% | 44.38% | 7.05 pp | -107 | 42 | -2.55 |
| BTC Daily | xgb | XGBoost | 592 | 238 | 354 | 40.20% | 35.42% | 40.42% | 9.80 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 759 | 356 | 403 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 759 | 355 | 404 | 46.77% | 44.17% | 45.42% | 3.23 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 759 | 340 | 419 | 44.80% | 44.58% | 44.38% | 5.20 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 759 | 339 | 420 | 44.66% | 41.25% | 45.21% | 5.34 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 759 | 334 | 425 | 44.01% | 42.92% | 45.42% | 5.99 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 759 | 326 | 433 | 42.95% | 41.67% | 44.38% | 7.05 pp | -107 | 42 | -2.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 582 | 288 | 294 | 49.48% | 52.08% | 49.79% | 0.52 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 582 | 287 | 295 | 49.31% | 49.17% | 49.79% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | nn | NN | 582 | 275 | 307 | 47.25% | 45.42% | 48.12% | 2.75 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 582 | 262 | 320 | 45.02% | 45.83% | 45.00% | 4.98 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 582 | 252 | 330 | 43.30% | 44.58% | 43.96% | 6.70 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 592 | 238 | 354 | 40.20% | 35.42% | 40.42% | 9.80 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 354 | 175 | 179 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 354 | 167 | 187 | 47.18% | 46.25% | 47.18% | 2.82 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 354 | 162 | 192 | 45.76% | 47.92% | 45.76% | 4.24 pp | -30 | 37 | -0.81 |
| BTC Market Hours | lstm | LSTM | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 354 | 152 | 202 | 42.94% | 42.50% | 42.94% | 7.06 pp | -50 | 37 | -1.35 |
| BTC Market Hours | xgb | XGBoost | 354 | 147 | 207 | 41.53% | 42.08% | 41.53% | 8.47 pp | -60 | 37 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 407 | 187 | 220 | 45.95% | 47.50% | 45.95% | 4.05 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 407 | 186 | 221 | 45.70% | 45.83% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 407 | 186 | 221 | 45.70% | 47.50% | 45.70% | 4.30 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 407 | 168 | 239 | 41.28% | 40.00% | 41.28% | 8.72 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | lstm | LSTM | 407 | 163 | 244 | 40.05% | 38.33% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 407 | 163 | 244 | 40.05% | 37.50% | 40.05% | 9.95 pp | -81 | 36 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 3 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 3 | -2.33 |

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
