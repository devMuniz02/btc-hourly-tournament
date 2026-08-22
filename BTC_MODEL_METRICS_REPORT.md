# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T00:49:24.709604+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-21 23:00:00+00:00 | 1080 | 791 | 289 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-21 23:00:00+00:00 | 891 | 566 | 324 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-21 23:00:00+00:00 | 463 | 328 | 134 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-21 23:00:00+00:00 | 465 | 382 | 81 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 2 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 2 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 0 | 2 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 08:00:00+00:00 | 2 | 0 | 2 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 328 | 163 | 165 | 49.70% | 48.75% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Daily | transformer | Transformer | 556 | 274 | 282 | 49.28% | 52.92% | 49.38% | 0.72 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 556 | 272 | 284 | 48.92% | 49.17% | 48.96% | 1.08 pp | -12 | 36 | -0.33 |
| BTC Market Hours | transformer | Transformer | 328 | 154 | 174 | 46.95% | 47.08% | 46.95% | 3.05 pp | -20 | 35 | -0.57 |
| BTC Market Hours Daily | nn | NN | 382 | 176 | 206 | 46.07% | 48.33% | 46.07% | 3.93 pp | -30 | 35 | -0.86 |
| BTC Daily | nn | NN | 556 | 262 | 294 | 47.12% | 45.42% | 47.29% | 2.88 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 382 | 173 | 209 | 45.29% | 46.25% | 45.29% | 4.71 pp | -36 | 35 | -1.03 |
| BTC Market Hours | nn | NN | 328 | 146 | 182 | 44.51% | 46.67% | 44.51% | 5.49 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 382 | 172 | 210 | 45.03% | 45.42% | 45.03% | 4.97 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 328 | 144 | 184 | 43.90% | 45.00% | 43.90% | 6.10 pp | -40 | 35 | -1.14 |
| BTC Daily | lstm | LSTM | 556 | 253 | 303 | 45.50% | 46.25% | 45.42% | 4.50 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 328 | 135 | 193 | 41.16% | 41.67% | 41.16% | 8.84 pp | -58 | 35 | -1.66 |
| BTC Market Hours | xgb | XGBoost | 328 | 132 | 196 | 40.24% | 40.42% | 40.24% | 9.76 pp | -64 | 35 | -1.83 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 556 | 242 | 314 | 43.53% | 45.00% | 43.96% | 6.47 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 382 | 155 | 227 | 40.58% | 38.75% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 382 | 155 | 227 | 40.58% | 37.92% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 382 | 151 | 231 | 39.53% | 37.50% | 39.53% | 10.47 pp | -80 | 35 | -2.29 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| BTC Daily | xgb | XGBoost | 566 | 229 | 337 | 40.46% | 36.25% | 40.83% | 9.54 pp | -108 | 36 | -3.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 556 | 274 | 282 | 49.28% | 52.92% | 49.38% | 0.72 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 556 | 272 | 284 | 48.92% | 49.17% | 48.96% | 1.08 pp | -12 | 36 | -0.33 |
| BTC Daily | nn | NN | 556 | 262 | 294 | 47.12% | 45.42% | 47.29% | 2.88 pp | -32 | 36 | -0.89 |
| BTC Daily | lstm | LSTM | 556 | 253 | 303 | 45.50% | 46.25% | 45.42% | 4.50 pp | -50 | 36 | -1.39 |
| BTC Daily | rf | RandomForest | 556 | 242 | 314 | 43.53% | 45.00% | 43.96% | 6.47 pp | -72 | 36 | -2.00 |
| BTC Daily | xgb | XGBoost | 566 | 229 | 337 | 40.46% | 36.25% | 40.83% | 9.54 pp | -108 | 36 | -3.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 328 | 163 | 165 | 49.70% | 48.75% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Market Hours | transformer | Transformer | 328 | 154 | 174 | 46.95% | 47.08% | 46.95% | 3.05 pp | -20 | 35 | -0.57 |
| BTC Market Hours | nn | NN | 328 | 146 | 182 | 44.51% | 46.67% | 44.51% | 5.49 pp | -36 | 35 | -1.03 |
| BTC Market Hours | lstm | LSTM | 328 | 144 | 184 | 43.90% | 45.00% | 43.90% | 6.10 pp | -40 | 35 | -1.14 |
| BTC Market Hours | rf | RandomForest | 328 | 135 | 193 | 41.16% | 41.67% | 41.16% | 8.84 pp | -58 | 35 | -1.66 |
| BTC Market Hours | xgb | XGBoost | 328 | 132 | 196 | 40.24% | 40.42% | 40.24% | 9.76 pp | -64 | 35 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 382 | 176 | 206 | 46.07% | 48.33% | 46.07% | 3.93 pp | -30 | 35 | -0.86 |
| BTC Market Hours Daily | transformer | Transformer | 382 | 173 | 209 | 45.29% | 46.25% | 45.29% | 4.71 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 382 | 172 | 210 | 45.03% | 45.42% | 45.03% | 4.97 pp | -38 | 35 | -1.09 |
| BTC Market Hours Daily | lstm | LSTM | 382 | 155 | 227 | 40.58% | 38.75% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 382 | 155 | 227 | 40.58% | 37.92% | 40.58% | 9.42 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | xgb | XGBoost | 382 | 151 | 231 | 39.53% | 37.50% | 39.53% | 10.47 pp | -80 | 35 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |

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
