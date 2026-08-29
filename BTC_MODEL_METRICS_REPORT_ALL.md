# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T04:16:25.156869+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1120 | 832 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 995 | 630 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 585 | 392 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 587 | 446 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 392 | 193 | 199 | 49.23% | 48.33% | 49.23% | 0.77 pp | -6 | 40 | -0.15 |
| BTC Daily | transformer | Transformer | 620 | 305 | 315 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 620 | 302 | 318 | 48.71% | 47.08% | 50.00% | 1.29 pp | -16 | 39 | -0.41 |
| BTC Market Hours | nn | NN | 392 | 183 | 209 | 46.68% | 48.75% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 392 | 183 | 209 | 46.68% | 44.17% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 446 | 207 | 239 | 46.41% | 48.33% | 46.41% | 3.59 pp | -32 | 39 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 446 | 205 | 241 | 45.96% | 45.83% | 45.96% | 4.04 pp | -36 | 39 | -0.92 |
| BTC Daily | nn | NN | 620 | 291 | 329 | 46.94% | 43.33% | 48.96% | 3.06 pp | -38 | 39 | -0.97 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 798 | 377 | 421 | 47.24% | 45.00% | 47.29% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 798 | 376 | 422 | 47.12% | 45.00% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 446 | 202 | 244 | 45.29% | 45.83% | 45.29% | 4.71 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 392 | 170 | 222 | 43.37% | 43.33% | 43.37% | 6.63 pp | -52 | 40 | -1.30 |
| BTC Market Hours | rf | RandomForest | 392 | 167 | 225 | 42.60% | 40.83% | 42.60% | 7.40 pp | -58 | 40 | -1.45 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 620 | 276 | 344 | 44.52% | 43.33% | 44.38% | 5.48 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 392 | 160 | 232 | 40.82% | 39.17% | 40.82% | 9.18 pp | -72 | 40 | -1.80 |
| BTC Hourly | nn | NN | 798 | 359 | 439 | 44.99% | 41.25% | 45.42% | 5.01 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 798 | 355 | 443 | 44.49% | 43.33% | 44.17% | 5.51 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 446 | 182 | 264 | 40.81% | 40.00% | 40.81% | 9.19 pp | -82 | 39 | -2.10 |
| BTC Hourly | lstm | LSTM | 798 | 352 | 446 | 44.11% | 44.58% | 45.83% | 5.89 pp | -94 | 43 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 446 | 179 | 267 | 40.13% | 38.33% | 40.13% | 9.87 pp | -88 | 39 | -2.26 |
| BTC Daily | rf | RandomForest | 620 | 265 | 355 | 42.74% | 42.50% | 43.75% | 7.26 pp | -90 | 39 | -2.31 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 446 | 176 | 270 | 39.46% | 37.50% | 39.46% | 10.54 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 798 | 340 | 458 | 42.61% | 40.00% | 44.38% | 7.39 pp | -118 | 43 | -2.74 |
| BTC Daily | xgb | XGBoost | 630 | 249 | 381 | 39.52% | 32.50% | 40.00% | 10.48 pp | -132 | 39 | -3.38 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 798 | 377 | 421 | 47.24% | 45.00% | 47.29% | 2.76 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 798 | 376 | 422 | 47.12% | 45.00% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 798 | 359 | 439 | 44.99% | 41.25% | 45.42% | 5.01 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 798 | 355 | 443 | 44.49% | 43.33% | 44.17% | 5.51 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 798 | 352 | 446 | 44.11% | 44.58% | 45.83% | 5.89 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 798 | 340 | 458 | 42.61% | 40.00% | 44.38% | 7.39 pp | -118 | 43 | -2.74 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 620 | 305 | 315 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 620 | 302 | 318 | 48.71% | 47.08% | 50.00% | 1.29 pp | -16 | 39 | -0.41 |
| BTC Daily | nn | NN | 620 | 291 | 329 | 46.94% | 43.33% | 48.96% | 3.06 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 620 | 276 | 344 | 44.52% | 43.33% | 44.38% | 5.48 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 620 | 265 | 355 | 42.74% | 42.50% | 43.75% | 7.26 pp | -90 | 39 | -2.31 |
| BTC Daily | xgb | XGBoost | 630 | 249 | 381 | 39.52% | 32.50% | 40.00% | 10.48 pp | -132 | 39 | -3.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 392 | 193 | 199 | 49.23% | 48.33% | 49.23% | 0.77 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 392 | 183 | 209 | 46.68% | 48.75% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 392 | 183 | 209 | 46.68% | 44.17% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | lstm | LSTM | 392 | 170 | 222 | 43.37% | 43.33% | 43.37% | 6.63 pp | -52 | 40 | -1.30 |
| BTC Market Hours | rf | RandomForest | 392 | 167 | 225 | 42.60% | 40.83% | 42.60% | 7.40 pp | -58 | 40 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 392 | 160 | 232 | 40.82% | 39.17% | 40.82% | 9.18 pp | -72 | 40 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 446 | 207 | 239 | 46.41% | 48.33% | 46.41% | 3.59 pp | -32 | 39 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 446 | 205 | 241 | 45.96% | 45.83% | 45.96% | 4.04 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 446 | 202 | 244 | 45.29% | 45.83% | 45.29% | 4.71 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 446 | 182 | 264 | 40.81% | 40.00% | 40.81% | 9.19 pp | -82 | 39 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 446 | 179 | 267 | 40.13% | 38.33% | 40.13% | 9.87 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 446 | 176 | 270 | 39.46% | 37.50% | 39.46% | 10.54 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

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
