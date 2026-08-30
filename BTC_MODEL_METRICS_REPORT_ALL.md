# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T23:47:50.167275+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1154 | 866 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1030 | 665 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 644 | 427 | 216 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 646 | 481 | 163 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 80 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 80 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 0 | 80 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 0 | 80 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 427 | 210 | 217 | 49.18% | 46.25% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 655 | 319 | 336 | 48.70% | 46.67% | 49.58% | 1.30 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 655 | 317 | 338 | 48.40% | 46.25% | 49.38% | 1.60 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 427 | 201 | 226 | 47.07% | 50.00% | 47.07% | 2.93 pp | -25 | 42 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 481 | 222 | 259 | 46.15% | 47.08% | 46.25% | 3.85 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 427 | 194 | 233 | 45.43% | 40.83% | 45.43% | 4.57 pp | -39 | 42 | -0.93 |
| BTC Hourly | transformer | Transformer | 832 | 395 | 437 | 47.48% | 47.08% | 46.67% | 2.52 pp | -42 | 45 | -0.93 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 481 | 219 | 262 | 45.53% | 44.17% | 45.62% | 4.47 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 481 | 218 | 263 | 45.32% | 45.00% | 45.42% | 4.68 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 655 | 306 | 349 | 46.72% | 41.67% | 48.96% | 3.28 pp | -43 | 40 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 832 | 390 | 442 | 46.88% | 42.50% | 46.46% | 3.12 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 427 | 186 | 241 | 43.56% | 43.33% | 43.56% | 6.44 pp | -55 | 42 | -1.31 |
| BTC Market Hours | rf | RandomForest | 427 | 184 | 243 | 43.09% | 42.92% | 43.09% | 6.91 pp | -59 | 42 | -1.40 |
| BTC Hourly | nn | NN | 832 | 376 | 456 | 45.19% | 43.33% | 44.58% | 4.81 pp | -80 | 45 | -1.78 |
| BTC Daily | lstm | LSTM | 655 | 290 | 365 | 44.27% | 40.83% | 43.75% | 5.73 pp | -75 | 40 | -1.88 |
| BTC Hourly | rf | RandomForest | 832 | 372 | 460 | 44.71% | 42.92% | 44.17% | 5.29 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 481 | 199 | 282 | 41.37% | 42.08% | 41.25% | 8.63 pp | -83 | 42 | -1.98 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 427 | 169 | 258 | 39.58% | 37.08% | 39.58% | 10.42 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 481 | 195 | 286 | 40.54% | 39.17% | 40.62% | 9.46 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 655 | 280 | 375 | 42.75% | 40.83% | 43.75% | 7.25 pp | -95 | 40 | -2.38 |
| BTC Hourly | lstm | LSTM | 832 | 359 | 473 | 43.15% | 39.58% | 42.71% | 6.85 pp | -114 | 45 | -2.53 |
| BTC Market Hours Daily | xgb | XGBoost | 481 | 186 | 295 | 38.67% | 35.00% | 38.54% | 11.33 pp | -109 | 42 | -2.60 |
| BTC Hourly | xgb | XGBoost | 832 | 353 | 479 | 42.43% | 39.58% | 42.71% | 7.57 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 665 | 263 | 402 | 39.55% | 32.92% | 39.58% | 10.45 pp | -139 | 40 | -3.48 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 832 | 395 | 437 | 47.48% | 47.08% | 46.67% | 2.52 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 832 | 390 | 442 | 46.88% | 42.50% | 46.46% | 3.12 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 832 | 376 | 456 | 45.19% | 43.33% | 44.58% | 4.81 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 832 | 372 | 460 | 44.71% | 42.92% | 44.17% | 5.29 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 832 | 359 | 473 | 43.15% | 39.58% | 42.71% | 6.85 pp | -114 | 45 | -2.53 |
| BTC Hourly | xgb | XGBoost | 832 | 353 | 479 | 42.43% | 39.58% | 42.71% | 7.57 pp | -126 | 45 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 655 | 319 | 336 | 48.70% | 46.67% | 49.58% | 1.30 pp | -17 | 40 | -0.42 |
| BTC Daily | transformer | Transformer | 655 | 317 | 338 | 48.40% | 46.25% | 49.38% | 1.60 pp | -21 | 40 | -0.53 |
| BTC Daily | nn | NN | 655 | 306 | 349 | 46.72% | 41.67% | 48.96% | 3.28 pp | -43 | 40 | -1.07 |
| BTC Daily | lstm | LSTM | 655 | 290 | 365 | 44.27% | 40.83% | 43.75% | 5.73 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 655 | 280 | 375 | 42.75% | 40.83% | 43.75% | 7.25 pp | -95 | 40 | -2.38 |
| BTC Daily | xgb | XGBoost | 665 | 263 | 402 | 39.55% | 32.92% | 39.58% | 10.45 pp | -139 | 40 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 427 | 210 | 217 | 49.18% | 46.25% | 49.18% | 0.82 pp | -7 | 42 | -0.17 |
| BTC Market Hours | nn | NN | 427 | 201 | 226 | 47.07% | 50.00% | 47.07% | 2.93 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 427 | 194 | 233 | 45.43% | 40.83% | 45.43% | 4.57 pp | -39 | 42 | -0.93 |
| BTC Market Hours | lstm | LSTM | 427 | 186 | 241 | 43.56% | 43.33% | 43.56% | 6.44 pp | -55 | 42 | -1.31 |
| BTC Market Hours | rf | RandomForest | 427 | 184 | 243 | 43.09% | 42.92% | 43.09% | 6.91 pp | -59 | 42 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 427 | 169 | 258 | 39.58% | 37.08% | 39.58% | 10.42 pp | -89 | 42 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 481 | 222 | 259 | 46.15% | 47.08% | 46.25% | 3.85 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | nn | NN | 481 | 219 | 262 | 45.53% | 44.17% | 45.62% | 4.47 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 481 | 218 | 263 | 45.32% | 45.00% | 45.42% | 4.68 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 481 | 199 | 282 | 41.37% | 42.08% | 41.25% | 8.63 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 481 | 195 | 286 | 40.54% | 39.17% | 40.62% | 9.46 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 481 | 186 | 295 | 38.67% | 35.00% | 38.54% | 11.33 pp | -109 | 42 | -2.60 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

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
