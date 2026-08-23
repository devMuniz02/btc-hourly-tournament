# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T01:59:01.725638+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 1105 | 791 | 314 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 931 | 581 | 349 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 492 | 343 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 494 | 397 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 0 | 9 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 0 | 9 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 343 | 170 | 173 | 49.56% | 48.33% | 49.56% | 0.44 pp | -3 | 36 | -0.08 |
| BTC Daily | transformer | Transformer | 571 | 281 | 290 | 49.21% | 53.33% | 49.17% | 0.79 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 571 | 280 | 291 | 49.04% | 48.33% | 49.17% | 0.96 pp | -11 | 37 | -0.30 |
| Consolidated Hourly | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 343 | 162 | 181 | 47.23% | 46.67% | 47.23% | 2.77 pp | -19 | 36 | -0.53 |
| BTC Market Hours Daily | nn | NN | 397 | 184 | 213 | 46.35% | 48.33% | 46.35% | 3.65 pp | -29 | 36 | -0.81 |
| BTC Daily | nn | NN | 571 | 270 | 301 | 47.29% | 46.25% | 48.12% | 2.71 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 397 | 182 | 215 | 45.84% | 46.25% | 45.84% | 4.16 pp | -33 | 36 | -0.92 |
| BTC Market Hours | nn | NN | 343 | 155 | 188 | 45.19% | 47.08% | 45.19% | 4.81 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 397 | 180 | 217 | 45.34% | 45.00% | 45.34% | 4.66 pp | -37 | 36 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 343 | 149 | 194 | 43.44% | 44.17% | 43.44% | 6.56 pp | -45 | 36 | -1.25 |
| BTC Market Hours | rf | RandomForest | 343 | 146 | 197 | 42.57% | 43.33% | 42.57% | 7.43 pp | -51 | 36 | -1.42 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| BTC Daily | lstm | LSTM | 571 | 257 | 314 | 45.01% | 45.83% | 44.58% | 4.99 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 343 | 142 | 201 | 41.40% | 41.67% | 41.40% | 8.60 pp | -59 | 36 | -1.64 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 397 | 163 | 234 | 41.06% | 39.17% | 41.06% | 8.94 pp | -71 | 36 | -1.97 |
| BTC Daily | rf | RandomForest | 571 | 248 | 323 | 43.43% | 45.83% | 43.96% | 6.57 pp | -75 | 37 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 397 | 161 | 236 | 40.55% | 39.58% | 40.55% | 9.45 pp | -75 | 36 | -2.08 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 397 | 157 | 240 | 39.55% | 37.08% | 39.55% | 10.45 pp | -83 | 36 | -2.31 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 581 | 236 | 345 | 40.62% | 36.25% | 40.83% | 9.38 pp | -109 | 37 | -2.95 |

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
| Consolidated Hourly | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

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
