# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T20:07:50.660251+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 19:00:00+00:00 | 1099 | 790 | 309 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 19:00:00+00:00 | 923 | 578 | 344 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 19:00:00+00:00 | 484 | 340 | 143 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 19:00:00+00:00 | 486 | 394 | 90 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-19T01:00:00+00:00 | 6 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-19T01:00:00+00:00 | 6 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-19T01:00:00+00:00 | 6 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-19T01:00:00+00:00 | 7 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 340 | 169 | 171 | 49.71% | 47.92% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Daily | transformer | Transformer | 568 | 279 | 289 | 49.12% | 52.92% | 48.75% | 0.88 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 568 | 277 | 291 | 48.77% | 47.50% | 49.17% | 1.23 pp | -14 | 36 | -0.39 |
| BTC Market Hours | transformer | Transformer | 340 | 159 | 181 | 46.76% | 46.25% | 46.76% | 3.24 pp | -22 | 36 | -0.61 |
| BTC Daily | nn | NN | 568 | 270 | 298 | 47.54% | 46.67% | 48.33% | 2.46 pp | -28 | 36 | -0.78 |
| BTC Market Hours Daily | nn | NN | 394 | 183 | 211 | 46.45% | 48.33% | 46.45% | 3.55 pp | -28 | 35 | -0.80 |
| BTC Market Hours | nn | NN | 340 | 153 | 187 | 45.00% | 47.08% | 45.00% | 5.00 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 394 | 180 | 214 | 45.69% | 45.42% | 45.69% | 4.31 pp | -34 | 35 | -0.97 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 394 | 178 | 216 | 45.18% | 45.00% | 45.18% | 4.82 pp | -38 | 35 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 340 | 149 | 191 | 43.82% | 44.58% | 43.82% | 6.18 pp | -42 | 36 | -1.17 |
| BTC Daily | lstm | LSTM | 568 | 257 | 311 | 45.25% | 46.25% | 45.21% | 4.75 pp | -54 | 36 | -1.50 |
| BTC Market Hours | rf | RandomForest | 340 | 143 | 197 | 42.06% | 42.08% | 42.06% | 7.94 pp | -54 | 36 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 340 | 140 | 200 | 41.18% | 41.25% | 41.18% | 8.82 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 568 | 248 | 320 | 43.66% | 46.25% | 44.17% | 6.34 pp | -72 | 36 | -2.00 |
| Consolidated Hourly | nn | NN | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 2 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 394 | 161 | 233 | 40.86% | 38.33% | 40.86% | 9.14 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 394 | 155 | 239 | 39.34% | 36.67% | 39.34% | 10.66 pp | -84 | 35 | -2.40 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 578 | 235 | 343 | 40.66% | 36.25% | 40.62% | 9.34 pp | -108 | 36 | -3.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 568 | 279 | 289 | 49.12% | 52.92% | 48.75% | 0.88 pp | -10 | 36 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 568 | 277 | 291 | 48.77% | 47.50% | 49.17% | 1.23 pp | -14 | 36 | -0.39 |
| BTC Daily | nn | NN | 568 | 270 | 298 | 47.54% | 46.67% | 48.33% | 2.46 pp | -28 | 36 | -0.78 |
| BTC Daily | lstm | LSTM | 568 | 257 | 311 | 45.25% | 46.25% | 45.21% | 4.75 pp | -54 | 36 | -1.50 |
| BTC Daily | rf | RandomForest | 568 | 248 | 320 | 43.66% | 46.25% | 44.17% | 6.34 pp | -72 | 36 | -2.00 |
| BTC Daily | xgb | XGBoost | 578 | 235 | 343 | 40.66% | 36.25% | 40.62% | 9.34 pp | -108 | 36 | -3.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 340 | 169 | 171 | 49.71% | 47.92% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Market Hours | transformer | Transformer | 340 | 159 | 181 | 46.76% | 46.25% | 46.76% | 3.24 pp | -22 | 36 | -0.61 |
| BTC Market Hours | nn | NN | 340 | 153 | 187 | 45.00% | 47.08% | 45.00% | 5.00 pp | -34 | 36 | -0.94 |
| BTC Market Hours | lstm | LSTM | 340 | 149 | 191 | 43.82% | 44.58% | 43.82% | 6.18 pp | -42 | 36 | -1.17 |
| BTC Market Hours | rf | RandomForest | 340 | 143 | 197 | 42.06% | 42.08% | 42.06% | 7.94 pp | -54 | 36 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 340 | 140 | 200 | 41.18% | 41.25% | 41.18% | 8.82 pp | -60 | 36 | -1.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 394 | 183 | 211 | 46.45% | 48.33% | 46.45% | 3.55 pp | -28 | 35 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 394 | 180 | 214 | 45.69% | 45.42% | 45.69% | 4.31 pp | -34 | 35 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 394 | 178 | 216 | 45.18% | 45.00% | 45.18% | 4.82 pp | -38 | 35 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 394 | 161 | 233 | 40.86% | 38.33% | 40.86% | 9.14 pp | -72 | 35 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 35 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 394 | 155 | 239 | 39.34% | 36.67% | 39.34% | 10.66 pp | -84 | 35 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 2 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 2 | -2.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
