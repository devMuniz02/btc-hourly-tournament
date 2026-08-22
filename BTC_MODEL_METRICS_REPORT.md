# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T02:24:51.943512+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 01:00:00+00:00 | 1082 | 791 | 291 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 01:00:00+00:00 | 894 | 567 | 326 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 465 | 329 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 467 | 383 | 82 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 3 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 3 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 0 | 3 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 0 | 3 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 329 | 163 | 166 | 49.54% | 48.75% | 49.54% | 0.46 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 557 | 274 | 283 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 48.75% | 48.75% | 1.17 pp | -13 | 36 | -0.36 |
| BTC Market Hours | transformer | Transformer | 329 | 155 | 174 | 47.11% | 47.08% | 47.11% | 2.89 pp | -19 | 35 | -0.54 |
| BTC Market Hours Daily | nn | NN | 383 | 176 | 207 | 45.95% | 48.33% | 45.95% | 4.05 pp | -31 | 35 | -0.89 |
| BTC Daily | nn | NN | 557 | 262 | 295 | 47.04% | 45.00% | 47.08% | 2.96 pp | -33 | 36 | -0.92 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 383 | 173 | 210 | 45.17% | 46.25% | 45.17% | 4.83 pp | -37 | 35 | -1.06 |
| BTC Market Hours | nn | NN | 329 | 146 | 183 | 44.38% | 46.67% | 44.38% | 5.62 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 383 | 172 | 211 | 44.91% | 45.42% | 44.91% | 5.09 pp | -39 | 35 | -1.11 |
| BTC Market Hours | lstm | LSTM | 329 | 145 | 184 | 44.07% | 45.42% | 44.07% | 5.93 pp | -39 | 35 | -1.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Daily | lstm | LSTM | 557 | 253 | 304 | 45.42% | 45.83% | 45.21% | 4.58 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 329 | 136 | 193 | 41.34% | 42.08% | 41.34% | 8.66 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 329 | 133 | 196 | 40.43% | 40.42% | 40.43% | 9.57 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 557 | 242 | 315 | 43.45% | 45.00% | 43.96% | 6.55 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 383 | 155 | 228 | 40.47% | 38.75% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 383 | 155 | 228 | 40.47% | 37.92% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 383 | 151 | 232 | 39.43% | 37.50% | 39.43% | 10.57 pp | -81 | 35 | -2.31 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| BTC Daily | xgb | XGBoost | 567 | 229 | 338 | 40.39% | 36.25% | 40.62% | 9.61 pp | -109 | 36 | -3.03 |

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
| BTC Daily | transformer | Transformer | 557 | 274 | 283 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 557 | 272 | 285 | 48.83% | 48.75% | 48.75% | 1.17 pp | -13 | 36 | -0.36 |
| BTC Daily | nn | NN | 557 | 262 | 295 | 47.04% | 45.00% | 47.08% | 2.96 pp | -33 | 36 | -0.92 |
| BTC Daily | lstm | LSTM | 557 | 253 | 304 | 45.42% | 45.83% | 45.21% | 4.58 pp | -51 | 36 | -1.42 |
| BTC Daily | rf | RandomForest | 557 | 242 | 315 | 43.45% | 45.00% | 43.96% | 6.55 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 567 | 229 | 338 | 40.39% | 36.25% | 40.62% | 9.61 pp | -109 | 36 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 329 | 163 | 166 | 49.54% | 48.75% | 49.54% | 0.46 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 329 | 155 | 174 | 47.11% | 47.08% | 47.11% | 2.89 pp | -19 | 35 | -0.54 |
| BTC Market Hours | nn | NN | 329 | 146 | 183 | 44.38% | 46.67% | 44.38% | 5.62 pp | -37 | 35 | -1.06 |
| BTC Market Hours | lstm | LSTM | 329 | 145 | 184 | 44.07% | 45.42% | 44.07% | 5.93 pp | -39 | 35 | -1.11 |
| BTC Market Hours | rf | RandomForest | 329 | 136 | 193 | 41.34% | 42.08% | 41.34% | 8.66 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 329 | 133 | 196 | 40.43% | 40.42% | 40.43% | 9.57 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 383 | 176 | 207 | 45.95% | 48.33% | 45.95% | 4.05 pp | -31 | 35 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 383 | 173 | 210 | 45.17% | 46.25% | 45.17% | 4.83 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 383 | 172 | 211 | 44.91% | 45.42% | 44.91% | 5.09 pp | -39 | 35 | -1.11 |
| BTC Market Hours Daily | lstm | LSTM | 383 | 155 | 228 | 40.47% | 38.75% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 383 | 155 | 228 | 40.47% | 37.92% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 383 | 151 | 232 | 39.43% | 37.50% | 39.43% | 10.57 pp | -81 | 35 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |

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
