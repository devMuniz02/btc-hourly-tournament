# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T23:08:28.656098+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 828 | 291 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 992 | 627 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 22:00:00+00:00 | 580 | 389 | 190 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 22:00:00+00:00 | 581 | 442 | 137 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 389 | 192 | 197 | 49.36% | 47.92% | 49.36% | 0.64 pp | -5 | 39 | -0.13 |
| BTC Daily | transformer | Transformer | 617 | 304 | 313 | 49.27% | 49.17% | 50.42% | 0.73 pp | -9 | 39 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 617 | 302 | 315 | 48.95% | 47.92% | 50.21% | 1.05 pp | -13 | 39 | -0.33 |
| BTC Market Hours | transformer | Transformer | 389 | 182 | 207 | 46.79% | 44.58% | 46.79% | 3.21 pp | -25 | 39 | -0.64 |
| BTC Market Hours | nn | NN | 389 | 181 | 208 | 46.53% | 49.17% | 46.53% | 3.47 pp | -27 | 39 | -0.69 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Daily | nn | NN | 617 | 290 | 327 | 47.00% | 43.75% | 49.17% | 3.00 pp | -37 | 39 | -0.95 |
| BTC Hourly | transformer | Transformer | 794 | 375 | 419 | 47.23% | 44.58% | 46.67% | 2.77 pp | -44 | 43 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 794 | 374 | 420 | 47.10% | 44.58% | 46.88% | 2.90 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 389 | 168 | 221 | 43.19% | 43.33% | 43.19% | 6.81 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 389 | 166 | 223 | 42.67% | 40.42% | 42.67% | 7.33 pp | -57 | 39 | -1.46 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| BTC Daily | lstm | LSTM | 617 | 276 | 341 | 44.73% | 43.75% | 44.79% | 5.27 pp | -65 | 39 | -1.67 |
| BTC Hourly | nn | NN | 794 | 358 | 436 | 45.09% | 40.83% | 45.42% | 4.91 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 389 | 159 | 230 | 40.87% | 39.17% | 40.87% | 9.13 pp | -71 | 39 | -1.82 |
| BTC Hourly | rf | RandomForest | 794 | 353 | 441 | 44.46% | 42.50% | 43.96% | 5.54 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Hourly | lstm | LSTM | 794 | 350 | 444 | 44.08% | 44.17% | 45.62% | 5.92 pp | -94 | 43 | -2.19 |
| BTC Daily | rf | RandomForest | 617 | 264 | 353 | 42.79% | 42.50% | 43.54% | 7.21 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 794 | 338 | 456 | 42.57% | 39.58% | 44.17% | 7.43 pp | -118 | 43 | -2.74 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 627 | 249 | 378 | 39.71% | 33.33% | 40.21% | 10.29 pp | -129 | 39 | -3.31 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 794 | 375 | 419 | 47.23% | 44.58% | 46.67% | 2.77 pp | -44 | 43 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 794 | 374 | 420 | 47.10% | 44.58% | 46.88% | 2.90 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 794 | 358 | 436 | 45.09% | 40.83% | 45.42% | 4.91 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 794 | 353 | 441 | 44.46% | 42.50% | 43.96% | 5.54 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 794 | 350 | 444 | 44.08% | 44.17% | 45.62% | 5.92 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 794 | 338 | 456 | 42.57% | 39.58% | 44.17% | 7.43 pp | -118 | 43 | -2.74 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 617 | 304 | 313 | 49.27% | 49.17% | 50.42% | 0.73 pp | -9 | 39 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 617 | 302 | 315 | 48.95% | 47.92% | 50.21% | 1.05 pp | -13 | 39 | -0.33 |
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
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

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
