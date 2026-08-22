# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T14:09:27.905887+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 1094 | 791 | 303 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 913 | 574 | 338 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 474 | 336 | 137 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 13:00:00+00:00 | 476 | 390 | 84 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 336 | 167 | 169 | 49.70% | 48.33% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Daily | transformer | Transformer | 564 | 278 | 286 | 49.29% | 52.50% | 49.17% | 0.71 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 564 | 276 | 288 | 48.94% | 47.92% | 48.96% | 1.06 pp | -12 | 36 | -0.33 |
| BTC Market Hours | transformer | Transformer | 336 | 158 | 178 | 47.02% | 46.67% | 47.02% | 2.98 pp | -20 | 35 | -0.57 |
| BTC Market Hours Daily | nn | NN | 390 | 181 | 209 | 46.41% | 48.75% | 46.41% | 3.59 pp | -28 | 35 | -0.80 |
| BTC Daily | nn | NN | 564 | 267 | 297 | 47.34% | 45.42% | 47.71% | 2.66 pp | -30 | 36 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 390 | 178 | 212 | 45.64% | 45.42% | 45.64% | 4.36 pp | -34 | 35 | -0.97 |
| BTC Market Hours | nn | NN | 336 | 151 | 185 | 44.94% | 46.67% | 44.94% | 5.06 pp | -34 | 35 | -0.97 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 390 | 177 | 213 | 45.38% | 45.83% | 45.38% | 4.62 pp | -36 | 35 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 336 | 147 | 189 | 43.75% | 44.17% | 43.75% | 6.25 pp | -42 | 35 | -1.20 |
| BTC Daily | lstm | LSTM | 564 | 254 | 310 | 45.04% | 45.42% | 44.79% | 4.96 pp | -56 | 36 | -1.56 |
| BTC Market Hours | rf | RandomForest | 336 | 140 | 196 | 41.67% | 41.67% | 41.67% | 8.33 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 336 | 137 | 199 | 40.77% | 40.00% | 40.77% | 9.23 pp | -62 | 35 | -1.77 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 564 | 245 | 319 | 43.44% | 45.00% | 44.38% | 6.56 pp | -74 | 36 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 390 | 158 | 232 | 40.51% | 37.50% | 40.51% | 9.49 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 390 | 157 | 233 | 40.26% | 38.33% | 40.26% | 9.74 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 390 | 153 | 237 | 39.23% | 36.67% | 39.23% | 10.77 pp | -84 | 35 | -2.40 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 574 | 232 | 342 | 40.42% | 35.83% | 40.21% | 9.58 pp | -110 | 36 | -3.06 |

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
| BTC Daily | transformer | Transformer | 564 | 278 | 286 | 49.29% | 52.50% | 49.17% | 0.71 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 564 | 276 | 288 | 48.94% | 47.92% | 48.96% | 1.06 pp | -12 | 36 | -0.33 |
| BTC Daily | nn | NN | 564 | 267 | 297 | 47.34% | 45.42% | 47.71% | 2.66 pp | -30 | 36 | -0.83 |
| BTC Daily | lstm | LSTM | 564 | 254 | 310 | 45.04% | 45.42% | 44.79% | 4.96 pp | -56 | 36 | -1.56 |
| BTC Daily | rf | RandomForest | 564 | 245 | 319 | 43.44% | 45.00% | 44.38% | 6.56 pp | -74 | 36 | -2.06 |
| BTC Daily | xgb | XGBoost | 574 | 232 | 342 | 40.42% | 35.83% | 40.21% | 9.58 pp | -110 | 36 | -3.06 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 336 | 167 | 169 | 49.70% | 48.33% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Market Hours | transformer | Transformer | 336 | 158 | 178 | 47.02% | 46.67% | 47.02% | 2.98 pp | -20 | 35 | -0.57 |
| BTC Market Hours | nn | NN | 336 | 151 | 185 | 44.94% | 46.67% | 44.94% | 5.06 pp | -34 | 35 | -0.97 |
| BTC Market Hours | lstm | LSTM | 336 | 147 | 189 | 43.75% | 44.17% | 43.75% | 6.25 pp | -42 | 35 | -1.20 |
| BTC Market Hours | rf | RandomForest | 336 | 140 | 196 | 41.67% | 41.67% | 41.67% | 8.33 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 336 | 137 | 199 | 40.77% | 40.00% | 40.77% | 9.23 pp | -62 | 35 | -1.77 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 390 | 181 | 209 | 46.41% | 48.75% | 46.41% | 3.59 pp | -28 | 35 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 390 | 178 | 212 | 45.64% | 45.42% | 45.64% | 4.36 pp | -34 | 35 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 390 | 177 | 213 | 45.38% | 45.83% | 45.38% | 4.62 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 390 | 158 | 232 | 40.51% | 37.50% | 40.51% | 9.49 pp | -74 | 35 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 390 | 157 | 233 | 40.26% | 38.33% | 40.26% | 9.74 pp | -76 | 35 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 390 | 153 | 237 | 39.23% | 36.67% | 39.23% | 10.77 pp | -84 | 35 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

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
