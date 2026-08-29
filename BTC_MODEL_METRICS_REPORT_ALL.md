# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T00:14:49.196390+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 829 | 290 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 992 | 627 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 23:00:00+00:00 | 581 | 389 | 191 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 23:00:00+00:00 | 583 | 443 | 138 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 389 | 192 | 197 | 49.36% | 47.92% | 49.36% | 0.64 pp | -5 | 39 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 617 | 303 | 314 | 49.11% | 48.75% | 50.21% | 0.89 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 617 | 301 | 316 | 48.78% | 47.50% | 50.00% | 1.22 pp | -15 | 39 | -0.38 |
| BTC Market Hours | transformer | Transformer | 389 | 182 | 207 | 46.79% | 44.58% | 46.79% | 3.21 pp | -25 | 39 | -0.64 |
| BTC Market Hours | nn | NN | 389 | 181 | 208 | 46.53% | 49.17% | 46.53% | 3.47 pp | -27 | 39 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 443 | 205 | 238 | 46.28% | 48.33% | 46.28% | 3.72 pp | -33 | 39 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 443 | 204 | 239 | 46.05% | 46.25% | 46.05% | 3.95 pp | -35 | 39 | -0.90 |
| BTC Daily | nn | NN | 617 | 290 | 327 | 47.00% | 43.75% | 49.17% | 3.00 pp | -37 | 39 | -0.95 |
| BTC Hourly | transformer | Transformer | 795 | 376 | 419 | 47.30% | 45.00% | 46.67% | 2.70 pp | -43 | 43 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 795 | 374 | 421 | 47.04% | 44.58% | 46.67% | 2.96 pp | -47 | 43 | -1.09 |
| BTC Market Hours Daily | nn | NN | 443 | 200 | 243 | 45.15% | 45.83% | 45.15% | 4.85 pp | -43 | 39 | -1.10 |
| BTC Market Hours | lstm | LSTM | 389 | 168 | 221 | 43.19% | 43.33% | 43.19% | 6.81 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 389 | 166 | 223 | 42.67% | 40.42% | 42.67% | 7.33 pp | -57 | 39 | -1.46 |
| BTC Daily | lstm | LSTM | 617 | 276 | 341 | 44.73% | 43.75% | 44.79% | 5.27 pp | -65 | 39 | -1.67 |
| BTC Hourly | nn | NN | 795 | 359 | 436 | 45.16% | 41.25% | 45.42% | 4.84 pp | -77 | 43 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 389 | 159 | 230 | 40.87% | 39.17% | 40.87% | 9.13 pp | -71 | 39 | -1.82 |
| BTC Hourly | rf | RandomForest | 795 | 354 | 441 | 44.53% | 42.92% | 43.96% | 5.47 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 443 | 181 | 262 | 40.86% | 40.00% | 40.86% | 9.14 pp | -81 | 39 | -2.08 |
| BTC Hourly | lstm | LSTM | 795 | 351 | 444 | 44.15% | 44.58% | 45.62% | 5.85 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 617 | 264 | 353 | 42.79% | 42.50% | 43.54% | 7.21 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | lstm | LSTM | 443 | 177 | 266 | 39.95% | 37.92% | 39.95% | 10.05 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 443 | 174 | 269 | 39.28% | 37.50% | 39.28% | 10.72 pp | -95 | 39 | -2.44 |
| BTC Hourly | xgb | XGBoost | 795 | 339 | 456 | 42.64% | 40.00% | 44.17% | 7.36 pp | -117 | 43 | -2.72 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| BTC Daily | xgb | XGBoost | 627 | 249 | 378 | 39.71% | 33.33% | 40.21% | 10.29 pp | -129 | 39 | -3.31 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 795 | 376 | 419 | 47.30% | 45.00% | 46.67% | 2.70 pp | -43 | 43 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 795 | 374 | 421 | 47.04% | 44.58% | 46.67% | 2.96 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 795 | 359 | 436 | 45.16% | 41.25% | 45.42% | 4.84 pp | -77 | 43 | -1.79 |
| BTC Hourly | rf | RandomForest | 795 | 354 | 441 | 44.53% | 42.92% | 43.96% | 5.47 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 795 | 351 | 444 | 44.15% | 44.58% | 45.62% | 5.85 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 795 | 339 | 456 | 42.64% | 40.00% | 44.17% | 7.36 pp | -117 | 43 | -2.72 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 617 | 303 | 314 | 49.11% | 48.75% | 50.21% | 0.89 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 617 | 301 | 316 | 48.78% | 47.50% | 50.00% | 1.22 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 617 | 290 | 327 | 47.00% | 43.75% | 49.17% | 3.00 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 617 | 276 | 341 | 44.73% | 43.75% | 44.79% | 5.27 pp | -65 | 39 | -1.67 |
| BTC Daily | rf | RandomForest | 617 | 264 | 353 | 42.79% | 42.50% | 43.54% | 7.21 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 627 | 249 | 378 | 39.71% | 33.33% | 40.21% | 10.29 pp | -129 | 39 | -3.31 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 389 | 192 | 197 | 49.36% | 47.92% | 49.36% | 0.64 pp | -5 | 39 | -0.13 |
| BTC Market Hours | transformer | Transformer | 389 | 182 | 207 | 46.79% | 44.58% | 46.79% | 3.21 pp | -25 | 39 | -0.64 |
| BTC Market Hours | nn | NN | 389 | 181 | 208 | 46.53% | 49.17% | 46.53% | 3.47 pp | -27 | 39 | -0.69 |
| BTC Market Hours | lstm | LSTM | 389 | 168 | 221 | 43.19% | 43.33% | 43.19% | 6.81 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 389 | 166 | 223 | 42.67% | 40.42% | 42.67% | 7.33 pp | -57 | 39 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 389 | 159 | 230 | 40.87% | 39.17% | 40.87% | 9.13 pp | -71 | 39 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 443 | 205 | 238 | 46.28% | 48.33% | 46.28% | 3.72 pp | -33 | 39 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 443 | 204 | 239 | 46.05% | 46.25% | 46.05% | 3.95 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | nn | NN | 443 | 200 | 243 | 45.15% | 45.83% | 45.15% | 4.85 pp | -43 | 39 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 443 | 181 | 262 | 40.86% | 40.00% | 40.86% | 9.14 pp | -81 | 39 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 443 | 177 | 266 | 39.95% | 37.92% | 39.95% | 10.05 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 443 | 174 | 269 | 39.28% | 37.50% | 39.28% | 10.72 pp | -95 | 39 | -2.44 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

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
