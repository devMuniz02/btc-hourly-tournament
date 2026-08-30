# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T22:16:04.686897+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1152 | 864 | 288 | 0 |
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
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 210 | 215 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 317 | 336 | 48.55% | 45.83% | 49.58% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 653 | 317 | 336 | 48.55% | 46.25% | 49.79% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 425 | 201 | 224 | 47.29% | 50.42% | 47.29% | 2.71 pp | -23 | 42 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | nn | NN | 479 | 226 | 253 | 47.18% | 45.42% | 47.18% | 2.82 pp | -27 | 42 | -0.64 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 47.08% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 221 | 258 | 46.14% | 43.75% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 425 | 194 | 231 | 45.65% | 41.67% | 45.65% | 4.35 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 830 | 395 | 435 | 47.59% | 47.92% | 46.88% | 2.41 pp | -40 | 45 | -0.89 |
| BTC Daily | nn | NN | 653 | 304 | 349 | 46.55% | 41.25% | 48.54% | 3.45 pp | -45 | 40 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 425 | 186 | 239 | 43.76% | 43.75% | 43.76% | 6.24 pp | -53 | 42 | -1.26 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 212 | 267 | 44.26% | 43.75% | 44.26% | 5.74 pp | -55 | 42 | -1.31 |
| BTC Market Hours | rf | RandomForest | 425 | 184 | 241 | 43.29% | 43.75% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Daily | lstm | LSTM | 653 | 288 | 365 | 44.10% | 40.83% | 43.33% | 5.90 pp | -77 | 40 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 199 | 280 | 41.54% | 39.58% | 41.54% | 8.46 pp | -81 | 42 | -1.93 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Market Hours | xgb | XGBoost | 425 | 169 | 256 | 39.76% | 37.92% | 39.76% | 10.24 pp | -87 | 42 | -2.07 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 37.08% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| BTC Daily | rf | RandomForest | 653 | 278 | 375 | 42.57% | 40.83% | 43.33% | 7.43 pp | -97 | 40 | -2.42 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 663 | 262 | 401 | 39.52% | 32.50% | 39.79% | 10.48 pp | -139 | 40 | -3.48 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 830 | 395 | 435 | 47.59% | 47.92% | 46.88% | 2.41 pp | -40 | 45 | -0.89 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 317 | 336 | 48.55% | 45.83% | 49.58% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 653 | 317 | 336 | 48.55% | 46.25% | 49.79% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 653 | 304 | 349 | 46.55% | 41.25% | 48.54% | 3.45 pp | -45 | 40 | -1.12 |
| BTC Daily | lstm | LSTM | 653 | 288 | 365 | 44.10% | 40.83% | 43.33% | 5.90 pp | -77 | 40 | -1.93 |
| BTC Daily | rf | RandomForest | 653 | 278 | 375 | 42.57% | 40.83% | 43.33% | 7.43 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 663 | 262 | 401 | 39.52% | 32.50% | 39.79% | 10.48 pp | -139 | 40 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 210 | 215 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Market Hours | nn | NN | 425 | 201 | 224 | 47.29% | 50.42% | 47.29% | 2.71 pp | -23 | 42 | -0.55 |
| BTC Market Hours | transformer | Transformer | 425 | 194 | 231 | 45.65% | 41.67% | 45.65% | 4.35 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 425 | 186 | 239 | 43.76% | 43.75% | 43.76% | 6.24 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 425 | 184 | 241 | 43.29% | 43.75% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 425 | 169 | 256 | 39.76% | 37.92% | 39.76% | 10.24 pp | -87 | 42 | -2.07 |

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
