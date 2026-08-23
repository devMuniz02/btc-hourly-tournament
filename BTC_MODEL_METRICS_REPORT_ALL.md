# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T00:26:20.844631+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 1103 | 790 | 313 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 929 | 580 | 348 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 490 | 342 | 147 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 492 | 396 | 94 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 03:00:00+00:00 | 8 | 8 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 03:00:00+00:00 | 8 | 8 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 03:00:00+00:00 | 8 | 0 | 8 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 03:00:00+00:00 | 8 | 0 | 8 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 342 | 170 | 172 | 49.71% | 48.33% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Daily | transformer | Transformer | 570 | 280 | 290 | 49.12% | 53.33% | 48.96% | 0.88 pp | -10 | 37 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 570 | 279 | 291 | 48.95% | 48.33% | 49.17% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Market Hours | transformer | Transformer | 342 | 161 | 181 | 47.08% | 46.67% | 47.08% | 2.92 pp | -20 | 36 | -0.56 |
| BTC Market Hours Daily | nn | NN | 396 | 183 | 213 | 46.21% | 48.33% | 46.21% | 3.79 pp | -30 | 36 | -0.83 |
| BTC Daily | nn | NN | 570 | 269 | 301 | 47.19% | 46.25% | 48.12% | 2.81 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 396 | 181 | 215 | 45.71% | 45.83% | 45.71% | 4.29 pp | -34 | 36 | -0.94 |
| BTC Market Hours | nn | NN | 342 | 154 | 188 | 45.03% | 47.08% | 45.03% | 4.97 pp | -34 | 36 | -0.94 |
| Consolidated Hourly | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 396 | 179 | 217 | 45.20% | 44.58% | 45.20% | 4.80 pp | -38 | 36 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 342 | 149 | 193 | 43.57% | 44.17% | 43.57% | 6.43 pp | -44 | 36 | -1.22 |
| BTC Market Hours | rf | RandomForest | 342 | 145 | 197 | 42.40% | 42.92% | 42.40% | 7.60 pp | -52 | 36 | -1.44 |
| BTC Daily | lstm | LSTM | 570 | 257 | 313 | 45.09% | 45.83% | 44.79% | 4.91 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 342 | 141 | 201 | 41.23% | 41.25% | 41.23% | 8.77 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 396 | 162 | 234 | 40.91% | 38.75% | 40.91% | 9.09 pp | -72 | 36 | -2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| BTC Daily | rf | RandomForest | 570 | 247 | 323 | 43.33% | 45.83% | 43.75% | 6.67 pp | -76 | 37 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 396 | 161 | 235 | 40.66% | 39.58% | 40.66% | 9.34 pp | -74 | 36 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 396 | 156 | 240 | 39.39% | 36.67% | 39.39% | 10.61 pp | -84 | 36 | -2.33 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 580 | 235 | 345 | 40.52% | 35.83% | 40.62% | 9.48 pp | -110 | 37 | -2.97 |
| Consolidated Hourly | nn | NN | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 2 | -3.00 |

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
| BTC Daily | transformer | Transformer | 570 | 280 | 290 | 49.12% | 53.33% | 48.96% | 0.88 pp | -10 | 37 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 570 | 279 | 291 | 48.95% | 48.33% | 49.17% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 570 | 269 | 301 | 47.19% | 46.25% | 48.12% | 2.81 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 570 | 257 | 313 | 45.09% | 45.83% | 44.79% | 4.91 pp | -56 | 37 | -1.51 |
| BTC Daily | rf | RandomForest | 570 | 247 | 323 | 43.33% | 45.83% | 43.75% | 6.67 pp | -76 | 37 | -2.05 |
| BTC Daily | xgb | XGBoost | 580 | 235 | 345 | 40.52% | 35.83% | 40.62% | 9.48 pp | -110 | 37 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 342 | 170 | 172 | 49.71% | 48.33% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Market Hours | transformer | Transformer | 342 | 161 | 181 | 47.08% | 46.67% | 47.08% | 2.92 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 342 | 154 | 188 | 45.03% | 47.08% | 45.03% | 4.97 pp | -34 | 36 | -0.94 |
| BTC Market Hours | lstm | LSTM | 342 | 149 | 193 | 43.57% | 44.17% | 43.57% | 6.43 pp | -44 | 36 | -1.22 |
| BTC Market Hours | rf | RandomForest | 342 | 145 | 197 | 42.40% | 42.92% | 42.40% | 7.60 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 342 | 141 | 201 | 41.23% | 41.25% | 41.23% | 8.77 pp | -60 | 36 | -1.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 396 | 183 | 213 | 46.21% | 48.33% | 46.21% | 3.79 pp | -30 | 36 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 396 | 181 | 215 | 45.71% | 45.83% | 45.71% | 4.29 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 396 | 179 | 217 | 45.20% | 44.58% | 45.20% | 4.80 pp | -38 | 36 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 396 | 162 | 234 | 40.91% | 38.75% | 40.91% | 9.09 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 396 | 161 | 235 | 40.66% | 39.58% | 40.66% | 9.34 pp | -74 | 36 | -2.06 |
| BTC Market Hours Daily | xgb | XGBoost | 396 | 156 | 240 | 39.39% | 36.67% | 39.39% | 10.61 pp | -84 | 36 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| Consolidated Hourly | nn | NN | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 2 | -3.00 |

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
