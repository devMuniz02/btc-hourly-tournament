# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T08:56:04.797309+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 962 | 597 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 526 | 359 | 166 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 359 | 178 | 181 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Daily | transformer | Transformer | 587 | 290 | 297 | 49.40% | 51.67% | 49.79% | 0.60 pp | -7 | 37 | -0.19 |
| Consolidated Hourly | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 587 | 286 | 301 | 48.72% | 47.08% | 49.38% | 1.28 pp | -15 | 37 | -0.41 |
| BTC Market Hours | transformer | Transformer | 359 | 170 | 189 | 47.35% | 46.25% | 47.35% | 2.65 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 359 | 164 | 195 | 45.68% | 47.92% | 45.68% | 4.32 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | nn | NN | 413 | 190 | 223 | 46.00% | 47.08% | 46.00% | 4.00 pp | -33 | 37 | -0.89 |
| BTC Daily | nn | NN | 587 | 276 | 311 | 47.02% | 45.42% | 48.12% | 2.98 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 413 | 189 | 224 | 45.76% | 46.25% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 413 | 189 | 224 | 45.76% | 47.92% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 765 | 360 | 405 | 47.06% | 43.33% | 47.50% | 2.94 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 765 | 357 | 408 | 46.67% | 43.33% | 45.42% | 3.33 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 359 | 155 | 204 | 43.18% | 42.92% | 43.18% | 6.82 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 359 | 154 | 205 | 42.90% | 42.08% | 42.90% | 7.10 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 587 | 264 | 323 | 44.97% | 45.00% | 45.21% | 5.03 pp | -59 | 37 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 359 | 147 | 212 | 40.95% | 41.67% | 40.95% | 9.05 pp | -65 | 37 | -1.76 |
| BTC Hourly | rf | RandomForest | 765 | 342 | 423 | 44.71% | 44.58% | 44.38% | 5.29 pp | -81 | 42 | -1.93 |
| BTC Hourly | nn | NN | 765 | 341 | 424 | 44.58% | 40.42% | 45.21% | 5.42 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 413 | 169 | 244 | 40.92% | 40.00% | 40.92% | 9.08 pp | -75 | 37 | -2.03 |
| BTC Hourly | lstm | LSTM | 765 | 337 | 428 | 44.05% | 43.33% | 45.42% | 5.95 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 587 | 253 | 334 | 43.10% | 43.33% | 43.96% | 6.90 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 413 | 166 | 247 | 40.19% | 38.75% | 40.19% | 9.81 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 413 | 165 | 248 | 39.95% | 38.33% | 39.95% | 10.05 pp | -83 | 37 | -2.24 |
| BTC Hourly | xgb | XGBoost | 765 | 327 | 438 | 42.75% | 41.25% | 44.17% | 7.25 pp | -111 | 42 | -2.64 |
| Consolidated Hourly | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 597 | 240 | 357 | 40.20% | 35.83% | 40.62% | 9.80 pp | -117 | 37 | -3.16 |

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
| BTC Daily | transformer | Transformer | 587 | 290 | 297 | 49.40% | 51.67% | 49.79% | 0.60 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 587 | 286 | 301 | 48.72% | 47.08% | 49.38% | 1.28 pp | -15 | 37 | -0.41 |
| BTC Daily | nn | NN | 587 | 276 | 311 | 47.02% | 45.42% | 48.12% | 2.98 pp | -35 | 37 | -0.95 |
| BTC Daily | lstm | LSTM | 587 | 264 | 323 | 44.97% | 45.00% | 45.21% | 5.03 pp | -59 | 37 | -1.59 |
| BTC Daily | rf | RandomForest | 587 | 253 | 334 | 43.10% | 43.33% | 43.96% | 6.90 pp | -81 | 37 | -2.19 |
| BTC Daily | xgb | XGBoost | 597 | 240 | 357 | 40.20% | 35.83% | 40.62% | 9.80 pp | -117 | 37 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 359 | 178 | 181 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Market Hours | transformer | Transformer | 359 | 170 | 189 | 47.35% | 46.25% | 47.35% | 2.65 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 359 | 164 | 195 | 45.68% | 47.92% | 45.68% | 4.32 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 359 | 155 | 204 | 43.18% | 42.92% | 43.18% | 6.82 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 359 | 154 | 205 | 42.90% | 42.08% | 42.90% | 7.10 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 359 | 147 | 212 | 40.95% | 41.67% | 40.95% | 9.05 pp | -65 | 37 | -1.76 |

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
