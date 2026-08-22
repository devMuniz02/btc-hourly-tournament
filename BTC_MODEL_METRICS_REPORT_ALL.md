# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T22:55:26.720195+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 1101 | 790 | 311 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 927 | 580 | 346 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 488 | 342 | 145 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 489 | 395 | 92 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 7 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 7 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 0 | 7 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 0 | 7 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 342 | 170 | 172 | 49.71% | 48.33% | 49.71% | 0.29 pp | -2 | 36 | -0.06 |
| BTC Daily | transformer | Transformer | 570 | 280 | 290 | 49.12% | 53.33% | 48.96% | 0.88 pp | -10 | 37 | -0.27 |
| BTC Daily | mlp_sklearn | MLPClassifier | 570 | 279 | 291 | 48.95% | 48.33% | 49.17% | 1.05 pp | -12 | 37 | -0.32 |
| Consolidated Hourly | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 342 | 161 | 181 | 47.08% | 46.67% | 47.08% | 2.92 pp | -20 | 36 | -0.56 |
| BTC Market Hours Daily | nn | NN | 395 | 183 | 212 | 46.33% | 48.33% | 46.33% | 3.67 pp | -29 | 36 | -0.81 |
| BTC Daily | nn | NN | 570 | 270 | 300 | 47.37% | 46.67% | 48.33% | 2.63 pp | -30 | 37 | -0.81 |
| BTC Market Hours | nn | NN | 342 | 154 | 188 | 45.03% | 47.08% | 45.03% | 4.97 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 395 | 180 | 215 | 45.57% | 45.42% | 45.57% | 4.43 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 395 | 178 | 217 | 45.06% | 44.58% | 45.06% | 4.94 pp | -39 | 36 | -1.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 342 | 149 | 193 | 43.57% | 44.17% | 43.57% | 6.43 pp | -44 | 36 | -1.22 |
| BTC Market Hours | rf | RandomForest | 342 | 145 | 197 | 42.40% | 42.92% | 42.40% | 7.60 pp | -52 | 36 | -1.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| BTC Daily | lstm | LSTM | 570 | 257 | 313 | 45.09% | 45.83% | 44.79% | 4.91 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 342 | 141 | 201 | 41.23% | 41.25% | 41.23% | 8.77 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 570 | 248 | 322 | 43.51% | 46.25% | 43.96% | 6.49 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 395 | 161 | 234 | 40.76% | 38.33% | 40.76% | 9.24 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 395 | 160 | 235 | 40.51% | 39.17% | 40.51% | 9.49 pp | -75 | 36 | -2.08 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 395 | 156 | 239 | 39.49% | 37.08% | 39.49% | 10.51 pp | -83 | 36 | -2.31 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |
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
| BTC Market Hours Daily | nn | NN | 395 | 183 | 212 | 46.33% | 48.33% | 46.33% | 3.67 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 395 | 180 | 215 | 45.57% | 45.42% | 45.57% | 4.43 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 395 | 178 | 217 | 45.06% | 44.58% | 45.06% | 4.94 pp | -39 | 36 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 395 | 161 | 234 | 40.76% | 38.33% | 40.76% | 9.24 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 395 | 160 | 235 | 40.51% | 39.17% | 40.51% | 9.49 pp | -75 | 36 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 395 | 156 | 239 | 39.49% | 37.08% | 39.49% | 10.51 pp | -83 | 36 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |

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
