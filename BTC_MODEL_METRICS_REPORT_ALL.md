# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T00:53:58.332840+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 930 | 581 | 348 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 491 | 343 | 147 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 23:00:00+00:00 | 493 | 397 | 94 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 343 | 170 | 173 | 49.56% | 48.33% | 49.56% | 0.44 pp | -3 | 36 | -0.08 |
| BTC Daily | transformer | Transformer | 571 | 281 | 290 | 49.21% | 53.33% | 49.17% | 0.79 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 571 | 280 | 291 | 49.04% | 48.33% | 49.17% | 0.96 pp | -11 | 37 | -0.30 |
| BTC Market Hours | transformer | Transformer | 343 | 162 | 181 | 47.23% | 46.67% | 47.23% | 2.77 pp | -19 | 36 | -0.53 |
| BTC Market Hours Daily | nn | NN | 397 | 184 | 213 | 46.35% | 48.33% | 46.35% | 3.65 pp | -29 | 36 | -0.81 |
| BTC Daily | nn | NN | 571 | 270 | 301 | 47.29% | 46.25% | 48.12% | 2.71 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 397 | 182 | 215 | 45.84% | 46.25% | 45.84% | 4.16 pp | -33 | 36 | -0.92 |
| BTC Market Hours | nn | NN | 343 | 155 | 188 | 45.19% | 47.08% | 45.19% | 4.81 pp | -33 | 36 | -0.92 |
| Consolidated Hourly | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 397 | 180 | 217 | 45.34% | 45.00% | 45.34% | 4.66 pp | -37 | 36 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 343 | 149 | 194 | 43.44% | 44.17% | 43.44% | 6.56 pp | -45 | 36 | -1.25 |
| BTC Market Hours | rf | RandomForest | 343 | 146 | 197 | 42.57% | 43.33% | 42.57% | 7.43 pp | -51 | 36 | -1.42 |
| BTC Daily | lstm | LSTM | 571 | 257 | 314 | 45.01% | 45.83% | 44.58% | 4.99 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 343 | 142 | 201 | 41.40% | 41.67% | 41.40% | 8.60 pp | -59 | 36 | -1.64 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 397 | 163 | 234 | 41.06% | 39.17% | 41.06% | 8.94 pp | -71 | 36 | -1.97 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 2 | -2.00 |
| BTC Daily | rf | RandomForest | 571 | 248 | 323 | 43.43% | 45.83% | 43.96% | 6.57 pp | -75 | 37 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 397 | 161 | 236 | 40.55% | 39.58% | 40.55% | 9.45 pp | -75 | 36 | -2.08 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 397 | 157 | 240 | 39.55% | 37.08% | 39.55% | 10.45 pp | -83 | 36 | -2.31 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 581 | 236 | 345 | 40.62% | 36.25% | 40.83% | 9.38 pp | -109 | 37 | -2.95 |
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
| BTC Daily | transformer | Transformer | 571 | 281 | 290 | 49.21% | 53.33% | 49.17% | 0.79 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 571 | 280 | 291 | 49.04% | 48.33% | 49.17% | 0.96 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 571 | 270 | 301 | 47.29% | 46.25% | 48.12% | 2.71 pp | -31 | 37 | -0.84 |
| BTC Daily | lstm | LSTM | 571 | 257 | 314 | 45.01% | 45.83% | 44.58% | 4.99 pp | -57 | 37 | -1.54 |
| BTC Daily | rf | RandomForest | 571 | 248 | 323 | 43.43% | 45.83% | 43.96% | 6.57 pp | -75 | 37 | -2.03 |
| BTC Daily | xgb | XGBoost | 581 | 236 | 345 | 40.62% | 36.25% | 40.83% | 9.38 pp | -109 | 37 | -2.95 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 343 | 170 | 173 | 49.56% | 48.33% | 49.56% | 0.44 pp | -3 | 36 | -0.08 |
| BTC Market Hours | transformer | Transformer | 343 | 162 | 181 | 47.23% | 46.67% | 47.23% | 2.77 pp | -19 | 36 | -0.53 |
| BTC Market Hours | nn | NN | 343 | 155 | 188 | 45.19% | 47.08% | 45.19% | 4.81 pp | -33 | 36 | -0.92 |
| BTC Market Hours | lstm | LSTM | 343 | 149 | 194 | 43.44% | 44.17% | 43.44% | 6.56 pp | -45 | 36 | -1.25 |
| BTC Market Hours | rf | RandomForest | 343 | 146 | 197 | 42.57% | 43.33% | 42.57% | 7.43 pp | -51 | 36 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 343 | 142 | 201 | 41.40% | 41.67% | 41.40% | 8.60 pp | -59 | 36 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 397 | 184 | 213 | 46.35% | 48.33% | 46.35% | 3.65 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 397 | 182 | 215 | 45.84% | 46.25% | 45.84% | 4.16 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 397 | 180 | 217 | 45.34% | 45.00% | 45.34% | 4.66 pp | -37 | 36 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 397 | 163 | 234 | 41.06% | 39.17% | 41.06% | 8.94 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | lstm | LSTM | 397 | 161 | 236 | 40.55% | 39.58% | 40.55% | 9.45 pp | -75 | 36 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 397 | 157 | 240 | 39.55% | 37.08% | 39.55% | 10.45 pp | -83 | 36 | -2.31 |

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
