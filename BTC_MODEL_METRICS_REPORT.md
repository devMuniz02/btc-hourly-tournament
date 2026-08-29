# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T07:55:31.137714+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1123 | 835 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 999 | 634 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 589 | 396 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 590 | 449 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 04:00:00+00:00 | 53 | 53 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 04:00:00+00:00 | 53 | 53 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 04:00:00+00:00 | 53 | 0 | 53 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 04:00:00+00:00 | 53 | 0 | 53 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 396 | 195 | 201 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Daily | transformer | Transformer | 624 | 306 | 318 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 624 | 305 | 319 | 48.88% | 46.67% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 396 | 186 | 210 | 46.97% | 49.58% | 46.97% | 3.03 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 396 | 184 | 212 | 46.46% | 42.92% | 46.46% | 3.54 pp | -28 | 40 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 449 | 207 | 242 | 46.10% | 47.92% | 46.10% | 3.90 pp | -35 | 40 | -0.88 |
| BTC Daily | nn | NN | 624 | 294 | 330 | 47.12% | 43.75% | 49.17% | 2.88 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 449 | 206 | 243 | 45.88% | 45.00% | 45.88% | 4.12 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 449 | 203 | 246 | 45.21% | 45.00% | 45.21% | 4.79 pp | -43 | 40 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 801 | 377 | 424 | 47.07% | 44.58% | 46.88% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | transformer | Transformer | 801 | 377 | 424 | 47.07% | 44.58% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| BTC Market Hours | lstm | LSTM | 396 | 173 | 223 | 43.69% | 43.75% | 43.69% | 6.31 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 396 | 168 | 228 | 42.42% | 40.83% | 42.42% | 7.58 pp | -60 | 40 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 624 | 278 | 346 | 44.55% | 42.92% | 44.38% | 5.45 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 396 | 161 | 235 | 40.66% | 39.17% | 40.66% | 9.34 pp | -74 | 40 | -1.85 |
| BTC Hourly | nn | NN | 801 | 359 | 442 | 44.82% | 40.42% | 45.00% | 5.18 pp | -83 | 43 | -1.93 |
| BTC Hourly | rf | RandomForest | 801 | 356 | 445 | 44.44% | 43.33% | 44.17% | 5.56 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 449 | 182 | 267 | 40.53% | 39.17% | 40.53% | 9.47 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 449 | 181 | 268 | 40.31% | 38.75% | 40.31% | 9.69 pp | -87 | 40 | -2.17 |
| BTC Hourly | lstm | LSTM | 801 | 352 | 449 | 43.95% | 43.75% | 45.21% | 6.05 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 624 | 268 | 356 | 42.95% | 42.92% | 43.75% | 7.05 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 449 | 176 | 273 | 39.20% | 37.08% | 39.20% | 10.80 pp | -97 | 40 | -2.42 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |
| BTC Hourly | xgb | XGBoost | 801 | 340 | 461 | 42.45% | 39.58% | 43.75% | 7.55 pp | -121 | 43 | -2.81 |
| BTC Daily | xgb | XGBoost | 634 | 250 | 384 | 39.43% | 32.50% | 40.00% | 10.57 pp | -134 | 39 | -3.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 801 | 377 | 424 | 47.07% | 44.58% | 46.88% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | transformer | Transformer | 801 | 377 | 424 | 47.07% | 44.58% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 801 | 359 | 442 | 44.82% | 40.42% | 45.00% | 5.18 pp | -83 | 43 | -1.93 |
| BTC Hourly | rf | RandomForest | 801 | 356 | 445 | 44.44% | 43.33% | 44.17% | 5.56 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 801 | 352 | 449 | 43.95% | 43.75% | 45.21% | 6.05 pp | -97 | 43 | -2.26 |
| BTC Hourly | xgb | XGBoost | 801 | 340 | 461 | 42.45% | 39.58% | 43.75% | 7.55 pp | -121 | 43 | -2.81 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 624 | 306 | 318 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 624 | 305 | 319 | 48.88% | 46.67% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 624 | 294 | 330 | 47.12% | 43.75% | 49.17% | 2.88 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 624 | 278 | 346 | 44.55% | 42.92% | 44.38% | 5.45 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 624 | 268 | 356 | 42.95% | 42.92% | 43.75% | 7.05 pp | -88 | 39 | -2.26 |
| BTC Daily | xgb | XGBoost | 634 | 250 | 384 | 39.43% | 32.50% | 40.00% | 10.57 pp | -134 | 39 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 396 | 195 | 201 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 396 | 186 | 210 | 46.97% | 49.58% | 46.97% | 3.03 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 396 | 184 | 212 | 46.46% | 42.92% | 46.46% | 3.54 pp | -28 | 40 | -0.70 |
| BTC Market Hours | lstm | LSTM | 396 | 173 | 223 | 43.69% | 43.75% | 43.69% | 6.31 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 396 | 168 | 228 | 42.42% | 40.83% | 42.42% | 7.58 pp | -60 | 40 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 396 | 161 | 235 | 40.66% | 39.17% | 40.66% | 9.34 pp | -74 | 40 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 449 | 207 | 242 | 46.10% | 47.92% | 46.10% | 3.90 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 449 | 206 | 243 | 45.88% | 45.00% | 45.88% | 4.12 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 449 | 203 | 246 | 45.21% | 45.00% | 45.21% | 4.79 pp | -43 | 40 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 449 | 182 | 267 | 40.53% | 39.17% | 40.53% | 9.47 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 449 | 181 | 268 | 40.31% | 38.75% | 40.31% | 9.69 pp | -87 | 40 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 449 | 176 | 273 | 39.20% | 37.08% | 39.20% | 10.80 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Hourly | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 53 | 30 | 23 | 56.60% | 56.60% | 56.60% | 6.60 pp | 7 | 6 | 1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 53 | 23 | 30 | 43.40% | 43.40% | 43.40% | 6.60 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 6 | -2.50 |

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
