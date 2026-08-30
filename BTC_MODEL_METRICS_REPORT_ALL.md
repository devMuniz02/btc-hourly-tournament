# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T22:23:42.339255+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1153 | 865 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1028 | 663 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 641 | 425 | 215 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 643 | 479 | 162 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 339 | 314 | 51.91% | 49.17% | 52.92% | 1.91 pp | 25 | 40 | 0.62 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 218 | 207 | 51.29% | 48.75% | 51.29% | 1.29 pp | 11 | 42 | 0.26 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| BTC Market Hours | nn | NN | 425 | 211 | 214 | 49.65% | 51.25% | 49.65% | 0.35 pp | -3 | 42 | -0.07 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | nn | NN | 479 | 226 | 253 | 47.18% | 45.42% | 47.18% | 2.82 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 653 | 312 | 341 | 47.78% | 44.58% | 48.96% | 2.22 pp | -29 | 40 | -0.72 |
| BTC Daily | transformer | Transformer | 653 | 310 | 343 | 47.47% | 45.00% | 49.17% | 2.53 pp | -33 | 40 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 47.08% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 221 | 258 | 46.14% | 43.75% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 831 | 395 | 436 | 47.53% | 47.50% | 46.88% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Market Hours | transformer | Transformer | 425 | 192 | 233 | 45.18% | 40.83% | 45.18% | 4.82 pp | -41 | 42 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| BTC Market Hours | rf | RandomForest | 425 | 188 | 237 | 44.24% | 44.58% | 44.24% | 5.76 pp | -49 | 42 | -1.17 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 389 | 442 | 46.81% | 42.50% | 46.46% | 3.19 pp | -53 | 45 | -1.18 |
| BTC Daily | lstm | LSTM | 653 | 301 | 352 | 46.09% | 41.25% | 45.62% | 3.91 pp | -51 | 40 | -1.27 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 212 | 267 | 44.26% | 43.75% | 44.26% | 5.74 pp | -55 | 42 | -1.31 |
| BTC Market Hours | lstm | LSTM | 425 | 184 | 241 | 43.29% | 42.08% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Daily | rf | RandomForest | 653 | 294 | 359 | 45.02% | 42.50% | 45.42% | 4.98 pp | -65 | 40 | -1.62 |
| BTC Hourly | nn | NN | 831 | 375 | 456 | 45.13% | 42.92% | 44.58% | 4.87 pp | -81 | 45 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 199 | 280 | 41.54% | 39.58% | 41.54% | 8.46 pp | -81 | 42 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 425 | 171 | 254 | 40.24% | 38.75% | 40.24% | 9.76 pp | -83 | 42 | -1.98 |
| BTC Hourly | rf | RandomForest | 831 | 371 | 460 | 44.65% | 42.92% | 44.17% | 5.35 pp | -89 | 45 | -1.98 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 37.08% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| BTC Hourly | lstm | LSTM | 831 | 359 | 472 | 43.20% | 40.00% | 42.92% | 6.80 pp | -113 | 45 | -2.51 |
| BTC Hourly | xgb | XGBoost | 831 | 352 | 479 | 42.36% | 39.17% | 42.71% | 7.64 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 663 | 266 | 397 | 40.12% | 34.17% | 40.62% | 9.88 pp | -131 | 40 | -3.27 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 831 | 395 | 436 | 47.53% | 47.50% | 46.88% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 831 | 389 | 442 | 46.81% | 42.50% | 46.46% | 3.19 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 831 | 375 | 456 | 45.13% | 42.92% | 44.58% | 4.87 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 831 | 371 | 460 | 44.65% | 42.92% | 44.17% | 5.35 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 831 | 359 | 472 | 43.20% | 40.00% | 42.92% | 6.80 pp | -113 | 45 | -2.51 |
| BTC Hourly | xgb | XGBoost | 831 | 352 | 479 | 42.36% | 39.17% | 42.71% | 7.64 pp | -127 | 45 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 339 | 314 | 51.91% | 49.17% | 52.92% | 1.91 pp | 25 | 40 | 0.62 |
| BTC Daily | nn | NN | 653 | 312 | 341 | 47.78% | 44.58% | 48.96% | 2.22 pp | -29 | 40 | -0.72 |
| BTC Daily | transformer | Transformer | 653 | 310 | 343 | 47.47% | 45.00% | 49.17% | 2.53 pp | -33 | 40 | -0.82 |
| BTC Daily | lstm | LSTM | 653 | 301 | 352 | 46.09% | 41.25% | 45.62% | 3.91 pp | -51 | 40 | -1.27 |
| BTC Daily | rf | RandomForest | 653 | 294 | 359 | 45.02% | 42.50% | 45.42% | 4.98 pp | -65 | 40 | -1.62 |
| BTC Daily | xgb | XGBoost | 663 | 266 | 397 | 40.12% | 34.17% | 40.62% | 9.88 pp | -131 | 40 | -3.27 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 218 | 207 | 51.29% | 48.75% | 51.29% | 1.29 pp | 11 | 42 | 0.26 |
| BTC Market Hours | nn | NN | 425 | 211 | 214 | 49.65% | 51.25% | 49.65% | 0.35 pp | -3 | 42 | -0.07 |
| BTC Market Hours | transformer | Transformer | 425 | 192 | 233 | 45.18% | 40.83% | 45.18% | 4.82 pp | -41 | 42 | -0.98 |
| BTC Market Hours | rf | RandomForest | 425 | 188 | 237 | 44.24% | 44.58% | 44.24% | 5.76 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 425 | 184 | 241 | 43.29% | 42.08% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 425 | 171 | 254 | 40.24% | 38.75% | 40.24% | 9.76 pp | -83 | 42 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 479 | 226 | 253 | 47.18% | 45.42% | 47.18% | 2.82 pp | -27 | 42 | -0.64 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 47.08% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 221 | 258 | 46.14% | 43.75% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 212 | 267 | 44.26% | 43.75% | 44.26% | 5.74 pp | -55 | 42 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 199 | 280 | 41.54% | 39.58% | 41.54% | 8.46 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 37.08% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

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
