# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T23:44:43.571300+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 22:00:00+00:00 | 1102 | 790 | 312 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 22:00:00+00:00 | 928 | 580 | 347 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 22:00:00+00:00 | 489 | 342 | 146 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 22:00:00+00:00 | 491 | 396 | 93 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 14:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 14:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 14:00:00+00:00 | 9 | 1 | 8 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 14:00:00+00:00 | 9 | 1 | 8 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 342 | 170 | 172 | 49.71% | 48.33% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Daily | transformer | Transformer | 570 | 280 | 290 | 49.12% | 53.33% | 48.96% | 0.88 pp | -10 | 37 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 570 | 279 | 291 | 48.95% | 48.33% | 49.17% | 1.05 pp | -12 | 37 | -0.32 |
| Consolidated Hourly | transformer | Transformer | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 342 | 161 | 181 | 47.08% | 46.67% | 47.08% | 2.92 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 570 | 270 | 300 | 47.37% | 46.67% | 48.33% | 2.63 pp | -30 | 37 | -0.81 |
| BTC Market Hours Daily | nn | NN | 396 | 183 | 213 | 46.21% | 48.33% | 46.21% | 3.79 pp | -30 | 36 | -0.83 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 396 | 181 | 215 | 45.71% | 45.83% | 45.71% | 4.29 pp | -34 | 36 | -0.94 |
| BTC Market Hours | nn | NN | 342 | 154 | 188 | 45.03% | 47.08% | 45.03% | 4.97 pp | -34 | 36 | -0.94 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 396 | 179 | 217 | 45.20% | 44.58% | 45.20% | 4.80 pp | -38 | 36 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 342 | 149 | 193 | 43.57% | 44.17% | 43.57% | 6.43 pp | -44 | 36 | -1.22 |
| BTC Market Hours | rf | RandomForest | 342 | 145 | 197 | 42.40% | 42.92% | 42.40% | 7.60 pp | -52 | 36 | -1.44 |
| Consolidated Hourly | rf | RandomForest | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| BTC Daily | lstm | LSTM | 570 | 257 | 313 | 45.09% | 45.83% | 44.79% | 4.91 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 342 | 141 | 201 | 41.23% | 41.25% | 41.23% | 8.77 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 570 | 248 | 322 | 43.51% | 46.25% | 43.96% | 6.49 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 396 | 162 | 234 | 40.91% | 38.75% | 40.91% | 9.09 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 396 | 161 | 235 | 40.66% | 39.58% | 40.66% | 9.34 pp | -74 | 36 | -2.06 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 396 | 156 | 240 | 39.39% | 36.67% | 39.39% | 10.61 pp | -84 | 36 | -2.33 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 580 | 236 | 344 | 40.69% | 36.25% | 40.83% | 9.31 pp | -108 | 37 | -2.92 |

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
| BTC Daily | nn | NN | 570 | 270 | 300 | 47.37% | 46.67% | 48.33% | 2.63 pp | -30 | 37 | -0.81 |
| BTC Daily | lstm | LSTM | 570 | 257 | 313 | 45.09% | 45.83% | 44.79% | 4.91 pp | -56 | 37 | -1.51 |
| BTC Daily | rf | RandomForest | 570 | 248 | 322 | 43.51% | 46.25% | 43.96% | 6.49 pp | -74 | 37 | -2.00 |
| BTC Daily | xgb | XGBoost | 580 | 236 | 344 | 40.69% | 36.25% | 40.83% | 9.31 pp | -108 | 37 | -2.92 |

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
| Consolidated Hourly | lstm | LSTM | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | rf | RandomForest | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
