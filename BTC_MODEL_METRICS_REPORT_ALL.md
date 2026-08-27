# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T09:06:38.797102+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 799 | 320 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 963 | 598 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 527 | 360 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 528 | 413 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 07:00:00+00:00 | 23 | 23 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 07:00:00+00:00 | 23 | 23 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 07:00:00+00:00 | 23 | 0 | 23 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 07:00:00+00:00 | 23 | 0 | 23 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 360 | 178 | 182 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | transformer | Transformer | 588 | 291 | 297 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| Consolidated Hourly | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 588 | 287 | 301 | 48.81% | 47.50% | 49.58% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Market Hours | transformer | Transformer | 360 | 171 | 189 | 47.50% | 46.25% | 47.50% | 2.50 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 360 | 164 | 196 | 45.56% | 47.50% | 45.56% | 4.44 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 413 | 190 | 223 | 46.00% | 47.08% | 46.00% | 4.00 pp | -33 | 37 | -0.89 |
| BTC Daily | nn | NN | 588 | 277 | 311 | 47.11% | 45.42% | 48.12% | 2.89 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 413 | 189 | 224 | 45.76% | 46.25% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 413 | 189 | 224 | 45.76% | 47.92% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 765 | 360 | 405 | 47.06% | 43.33% | 47.50% | 2.94 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 765 | 357 | 408 | 46.67% | 43.33% | 45.42% | 3.33 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 360 | 156 | 204 | 43.33% | 43.33% | 43.33% | 6.67 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 360 | 154 | 206 | 42.78% | 41.67% | 42.78% | 7.22 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 588 | 264 | 324 | 44.90% | 45.00% | 45.00% | 5.10 pp | -60 | 37 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 360 | 147 | 213 | 40.83% | 41.67% | 40.83% | 9.17 pp | -66 | 37 | -1.78 |
| BTC Hourly | rf | RandomForest | 765 | 342 | 423 | 44.71% | 44.58% | 44.38% | 5.29 pp | -81 | 42 | -1.93 |
| BTC Hourly | nn | NN | 765 | 341 | 424 | 44.58% | 40.42% | 45.21% | 5.42 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 413 | 169 | 244 | 40.92% | 40.00% | 40.92% | 9.08 pp | -75 | 37 | -2.03 |
| BTC Daily | rf | RandomForest | 588 | 254 | 334 | 43.20% | 43.75% | 44.17% | 6.80 pp | -80 | 37 | -2.16 |
| BTC Hourly | lstm | LSTM | 765 | 337 | 428 | 44.05% | 43.33% | 45.42% | 5.95 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 413 | 166 | 247 | 40.19% | 38.75% | 40.19% | 9.81 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 413 | 165 | 248 | 39.95% | 38.33% | 39.95% | 10.05 pp | -83 | 37 | -2.24 |
| BTC Hourly | xgb | XGBoost | 765 | 327 | 438 | 42.75% | 41.25% | 44.17% | 7.25 pp | -111 | 42 | -2.64 |
| Consolidated Hourly | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 598 | 241 | 357 | 40.30% | 36.25% | 40.62% | 9.70 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 765 | 360 | 405 | 47.06% | 43.33% | 47.50% | 2.94 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 765 | 357 | 408 | 46.67% | 43.33% | 45.42% | 3.33 pp | -51 | 42 | -1.21 |
| BTC Hourly | rf | RandomForest | 765 | 342 | 423 | 44.71% | 44.58% | 44.38% | 5.29 pp | -81 | 42 | -1.93 |
| BTC Hourly | nn | NN | 765 | 341 | 424 | 44.58% | 40.42% | 45.21% | 5.42 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 765 | 337 | 428 | 44.05% | 43.33% | 45.42% | 5.95 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 765 | 327 | 438 | 42.75% | 41.25% | 44.17% | 7.25 pp | -111 | 42 | -2.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 588 | 291 | 297 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 588 | 287 | 301 | 48.81% | 47.50% | 49.58% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Daily | nn | NN | 588 | 277 | 311 | 47.11% | 45.42% | 48.12% | 2.89 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 588 | 264 | 324 | 44.90% | 45.00% | 45.00% | 5.10 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 588 | 254 | 334 | 43.20% | 43.75% | 44.17% | 6.80 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 598 | 241 | 357 | 40.30% | 36.25% | 40.62% | 9.70 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 360 | 178 | 182 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 360 | 171 | 189 | 47.50% | 46.25% | 47.50% | 2.50 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 360 | 164 | 196 | 45.56% | 47.50% | 45.56% | 4.44 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 360 | 156 | 204 | 43.33% | 43.33% | 43.33% | 6.67 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 360 | 154 | 206 | 42.78% | 41.67% | 42.78% | 7.22 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 360 | 147 | 213 | 40.83% | 41.67% | 40.83% | 9.17 pp | -66 | 37 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 413 | 190 | 223 | 46.00% | 47.08% | 46.00% | 4.00 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 413 | 189 | 224 | 45.76% | 46.25% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 413 | 189 | 224 | 45.76% | 47.92% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 413 | 169 | 244 | 40.92% | 40.00% | 40.92% | 9.08 pp | -75 | 37 | -2.03 |
| BTC Market Hours Daily | xgb | XGBoost | 413 | 166 | 247 | 40.19% | 38.75% | 40.19% | 9.81 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 413 | 165 | 248 | 39.95% | 38.33% | 39.95% | 10.05 pp | -83 | 37 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 23 | 15 | 8 | 65.22% | 65.22% | 65.22% | 15.22 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 23 | 12 | 11 | 52.17% | 52.17% | 52.17% | 2.17 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |

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
