# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T14:27:46.568230+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1128 | 840 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1004 | 639 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 13:00:00+00:00 | 596 | 401 | 194 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 13:00:00+00:00 | 597 | 454 | 141 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 58 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 58 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 0 | 58 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 0 | 58 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 401 | 199 | 202 | 49.63% | 48.33% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| BTC Daily | transformer | Transformer | 629 | 309 | 320 | 49.13% | 47.92% | 49.79% | 0.87 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 629 | 308 | 321 | 48.97% | 46.67% | 50.21% | 1.03 pp | -13 | 39 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| BTC Market Hours | nn | NN | 401 | 190 | 211 | 47.38% | 50.83% | 47.38% | 2.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 401 | 185 | 216 | 46.13% | 42.08% | 46.13% | 3.87 pp | -31 | 40 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 454 | 209 | 245 | 46.04% | 45.42% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 454 | 209 | 245 | 46.04% | 47.50% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Daily | nn | NN | 629 | 296 | 333 | 47.06% | 43.33% | 48.96% | 2.94 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | nn | NN | 454 | 207 | 247 | 45.59% | 46.25% | 45.59% | 4.41 pp | -40 | 40 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 806 | 380 | 426 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 806 | 380 | 426 | 47.15% | 44.58% | 46.46% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Market Hours | lstm | LSTM | 401 | 177 | 224 | 44.14% | 45.42% | 44.14% | 5.86 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 401 | 172 | 229 | 42.89% | 41.67% | 42.89% | 7.11 pp | -57 | 40 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 629 | 279 | 350 | 44.36% | 42.50% | 43.96% | 5.64 pp | -71 | 39 | -1.82 |
| BTC Hourly | nn | NN | 806 | 362 | 444 | 44.91% | 40.00% | 44.58% | 5.09 pp | -82 | 44 | -1.86 |
| BTC Hourly | rf | RandomForest | 806 | 360 | 446 | 44.67% | 43.75% | 44.38% | 5.33 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 37.92% | 40.15% | 9.85 pp | -79 | 40 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 454 | 185 | 269 | 40.75% | 40.00% | 40.75% | 9.25 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 454 | 184 | 270 | 40.53% | 39.17% | 40.53% | 9.47 pp | -86 | 40 | -2.15 |
| BTC Hourly | lstm | LSTM | 806 | 353 | 453 | 43.80% | 42.50% | 45.00% | 6.20 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 629 | 269 | 360 | 42.77% | 42.50% | 43.54% | 7.23 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 454 | 178 | 276 | 39.21% | 37.08% | 39.21% | 10.79 pp | -98 | 40 | -2.45 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 806 | 342 | 464 | 42.43% | 39.58% | 43.33% | 7.57 pp | -122 | 44 | -2.77 |
| BTC Daily | xgb | XGBoost | 639 | 251 | 388 | 39.28% | 31.67% | 39.38% | 10.72 pp | -137 | 39 | -3.51 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 806 | 380 | 426 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 806 | 380 | 426 | 47.15% | 44.58% | 46.46% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | nn | NN | 806 | 362 | 444 | 44.91% | 40.00% | 44.58% | 5.09 pp | -82 | 44 | -1.86 |
| BTC Hourly | rf | RandomForest | 806 | 360 | 446 | 44.67% | 43.75% | 44.38% | 5.33 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 806 | 353 | 453 | 43.80% | 42.50% | 45.00% | 6.20 pp | -100 | 44 | -2.27 |
| BTC Hourly | xgb | XGBoost | 806 | 342 | 464 | 42.43% | 39.58% | 43.33% | 7.57 pp | -122 | 44 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 629 | 309 | 320 | 49.13% | 47.92% | 49.79% | 0.87 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 629 | 308 | 321 | 48.97% | 46.67% | 50.21% | 1.03 pp | -13 | 39 | -0.33 |
| BTC Daily | nn | NN | 629 | 296 | 333 | 47.06% | 43.33% | 48.96% | 2.94 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 629 | 279 | 350 | 44.36% | 42.50% | 43.96% | 5.64 pp | -71 | 39 | -1.82 |
| BTC Daily | rf | RandomForest | 629 | 269 | 360 | 42.77% | 42.50% | 43.54% | 7.23 pp | -91 | 39 | -2.33 |
| BTC Daily | xgb | XGBoost | 639 | 251 | 388 | 39.28% | 31.67% | 39.38% | 10.72 pp | -137 | 39 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 401 | 199 | 202 | 49.63% | 48.33% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| BTC Market Hours | nn | NN | 401 | 190 | 211 | 47.38% | 50.83% | 47.38% | 2.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 401 | 185 | 216 | 46.13% | 42.08% | 46.13% | 3.87 pp | -31 | 40 | -0.78 |
| BTC Market Hours | lstm | LSTM | 401 | 177 | 224 | 44.14% | 45.42% | 44.14% | 5.86 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 401 | 172 | 229 | 42.89% | 41.67% | 42.89% | 7.11 pp | -57 | 40 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 37.92% | 40.15% | 9.85 pp | -79 | 40 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 454 | 209 | 245 | 46.04% | 45.42% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 454 | 209 | 245 | 46.04% | 47.50% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | nn | NN | 454 | 207 | 247 | 45.59% | 46.25% | 45.59% | 4.41 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 454 | 185 | 269 | 40.75% | 40.00% | 40.75% | 9.25 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 454 | 184 | 270 | 40.53% | 39.17% | 40.53% | 9.47 pp | -86 | 40 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 454 | 178 | 276 | 39.21% | 37.08% | 39.21% | 10.79 pp | -98 | 40 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

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
