# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T03:30:15.492885+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1157 | 869 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1032 | 667 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 648 | 429 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 650 | 483 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 429 | 211 | 218 | 49.18% | 46.67% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 657 | 321 | 336 | 48.86% | 47.08% | 49.79% | 1.14 pp | -15 | 40 | -0.38 |
| BTC Daily | transformer | Transformer | 657 | 317 | 340 | 48.25% | 45.83% | 49.38% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Market Hours | nn | NN | 429 | 202 | 227 | 47.09% | 50.00% | 47.09% | 2.91 pp | -25 | 42 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 222 | 261 | 45.96% | 47.08% | 46.25% | 4.04 pp | -39 | 42 | -0.93 |
| BTC Market Hours | transformer | Transformer | 429 | 195 | 234 | 45.45% | 40.83% | 45.45% | 4.55 pp | -39 | 42 | -0.93 |
| BTC Hourly | transformer | Transformer | 835 | 396 | 439 | 47.43% | 47.50% | 46.88% | 2.57 pp | -43 | 45 | -0.96 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 483 | 219 | 264 | 45.34% | 43.75% | 45.62% | 4.66 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 657 | 307 | 350 | 46.73% | 42.08% | 49.17% | 3.27 pp | -43 | 40 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 218 | 265 | 45.13% | 44.58% | 45.21% | 4.87 pp | -47 | 42 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 835 | 391 | 444 | 46.83% | 42.92% | 46.67% | 3.17 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 429 | 186 | 243 | 43.36% | 43.33% | 43.36% | 6.64 pp | -57 | 42 | -1.36 |
| BTC Market Hours | rf | RandomForest | 429 | 185 | 244 | 43.12% | 42.92% | 43.12% | 6.88 pp | -59 | 42 | -1.40 |
| BTC Hourly | nn | NN | 835 | 377 | 458 | 45.15% | 43.75% | 44.38% | 4.85 pp | -81 | 45 | -1.80 |
| BTC Daily | lstm | LSTM | 657 | 290 | 367 | 44.14% | 40.00% | 43.54% | 5.86 pp | -77 | 40 | -1.93 |
| BTC Hourly | rf | RandomForest | 835 | 373 | 462 | 44.67% | 43.33% | 43.96% | 5.33 pp | -89 | 45 | -1.98 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 199 | 284 | 41.20% | 41.67% | 41.25% | 8.80 pp | -85 | 42 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 429 | 170 | 259 | 39.63% | 37.50% | 39.63% | 10.37 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 195 | 288 | 40.37% | 39.17% | 40.42% | 9.63 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 657 | 282 | 375 | 42.92% | 41.25% | 44.17% | 7.08 pp | -93 | 40 | -2.33 |
| BTC Hourly | lstm | LSTM | 835 | 360 | 475 | 43.11% | 40.00% | 42.71% | 6.89 pp | -115 | 45 | -2.56 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 187 | 296 | 38.72% | 35.00% | 38.75% | 11.28 pp | -109 | 42 | -2.60 |
| BTC Hourly | xgb | XGBoost | 835 | 353 | 482 | 42.28% | 39.17% | 42.50% | 7.72 pp | -129 | 45 | -2.87 |
| BTC Daily | xgb | XGBoost | 667 | 265 | 402 | 39.73% | 33.33% | 40.00% | 10.27 pp | -137 | 40 | -3.42 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 835 | 396 | 439 | 47.43% | 47.50% | 46.88% | 2.57 pp | -43 | 45 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 835 | 391 | 444 | 46.83% | 42.92% | 46.67% | 3.17 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 835 | 377 | 458 | 45.15% | 43.75% | 44.38% | 4.85 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 835 | 373 | 462 | 44.67% | 43.33% | 43.96% | 5.33 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 835 | 360 | 475 | 43.11% | 40.00% | 42.71% | 6.89 pp | -115 | 45 | -2.56 |
| BTC Hourly | xgb | XGBoost | 835 | 353 | 482 | 42.28% | 39.17% | 42.50% | 7.72 pp | -129 | 45 | -2.87 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 657 | 321 | 336 | 48.86% | 47.08% | 49.79% | 1.14 pp | -15 | 40 | -0.38 |
| BTC Daily | transformer | Transformer | 657 | 317 | 340 | 48.25% | 45.83% | 49.38% | 1.75 pp | -23 | 40 | -0.57 |
| BTC Daily | nn | NN | 657 | 307 | 350 | 46.73% | 42.08% | 49.17% | 3.27 pp | -43 | 40 | -1.07 |
| BTC Daily | lstm | LSTM | 657 | 290 | 367 | 44.14% | 40.00% | 43.54% | 5.86 pp | -77 | 40 | -1.93 |
| BTC Daily | rf | RandomForest | 657 | 282 | 375 | 42.92% | 41.25% | 44.17% | 7.08 pp | -93 | 40 | -2.33 |
| BTC Daily | xgb | XGBoost | 667 | 265 | 402 | 39.73% | 33.33% | 40.00% | 10.27 pp | -137 | 40 | -3.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 429 | 211 | 218 | 49.18% | 46.67% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Market Hours | nn | NN | 429 | 202 | 227 | 47.09% | 50.00% | 47.09% | 2.91 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 429 | 195 | 234 | 45.45% | 40.83% | 45.45% | 4.55 pp | -39 | 42 | -0.93 |
| BTC Market Hours | lstm | LSTM | 429 | 186 | 243 | 43.36% | 43.33% | 43.36% | 6.64 pp | -57 | 42 | -1.36 |
| BTC Market Hours | rf | RandomForest | 429 | 185 | 244 | 43.12% | 42.92% | 43.12% | 6.88 pp | -59 | 42 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 429 | 170 | 259 | 39.63% | 37.50% | 39.63% | 10.37 pp | -89 | 42 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 483 | 222 | 261 | 45.96% | 47.08% | 46.25% | 4.04 pp | -39 | 42 | -0.93 |
| BTC Market Hours Daily | nn | NN | 483 | 219 | 264 | 45.34% | 43.75% | 45.62% | 4.66 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 483 | 218 | 265 | 45.13% | 44.58% | 45.21% | 4.87 pp | -47 | 42 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 483 | 199 | 284 | 41.20% | 41.67% | 41.25% | 8.80 pp | -85 | 42 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 483 | 195 | 288 | 40.37% | 39.17% | 40.42% | 9.63 pp | -93 | 42 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 483 | 187 | 296 | 38.72% | 35.00% | 38.75% | 11.28 pp | -109 | 42 | -2.60 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

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
