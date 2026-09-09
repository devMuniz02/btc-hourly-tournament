# BTC Model Metrics Report - All Rows

Generated at: 2026-09-09T01:50:48.277122+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1301 | 1013 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1177 | 812 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 910 | 574 | 335 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 912 | 628 | 282 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 216 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T17:00:00+00:00 | 217 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 574 | 280 | 294 | 48.78% | 47.92% | 48.12% | 1.22 pp | -14 | 54 | -0.26 |
| BTC Market Hours | nn | NN | 574 | 274 | 300 | 47.74% | 52.08% | 49.38% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 574 | 270 | 304 | 47.04% | 46.67% | 46.88% | 2.96 pp | -34 | 54 | -0.63 |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| Consolidated Hourly | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| BTC Market Hours Daily | nn | NN | 628 | 293 | 335 | 46.66% | 47.50% | 47.92% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 628 | 293 | 335 | 46.66% | 49.17% | 47.29% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 628 | 292 | 336 | 46.50% | 48.33% | 46.88% | 3.50 pp | -44 | 53 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| BTC Daily | nn | NN | 802 | 373 | 429 | 46.51% | 45.00% | 44.79% | 3.49 pp | -56 | 46 | -1.22 |
| Consolidated Hourly | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Market Hours | lstm | LSTM | 574 | 249 | 325 | 43.38% | 43.33% | 43.75% | 6.62 pp | -76 | 54 | -1.41 |
| Consolidated Hourly | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| BTC Market Hours | rf | RandomForest | 574 | 248 | 326 | 43.21% | 45.42% | 43.75% | 6.79 pp | -78 | 54 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 574 | 248 | 326 | 43.21% | 47.08% | 43.54% | 6.79 pp | -78 | 54 | -1.44 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 628 | 262 | 366 | 41.72% | 43.33% | 40.62% | 8.28 pp | -104 | 53 | -1.96 |
| Consolidated Hourly | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 628 | 260 | 368 | 41.40% | 44.58% | 41.04% | 8.60 pp | -108 | 53 | -2.04 |
| Consolidated Hourly | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |
| Consolidated Daily/Hourly Refresh | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 628 | 256 | 372 | 40.76% | 41.67% | 40.00% | 9.24 pp | -116 | 53 | -2.19 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| BTC Daily | lstm | LSTM | 802 | 337 | 465 | 42.02% | 34.58% | 40.00% | 7.98 pp | -128 | 46 | -2.78 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Daily | rf | RandomForest | 802 | 334 | 468 | 41.65% | 37.08% | 41.04% | 8.35 pp | -134 | 46 | -2.91 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |
| BTC Daily | xgb | XGBoost | 812 | 316 | 496 | 38.92% | 35.00% | 35.62% | 11.08 pp | -180 | 46 | -3.91 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 979 | 467 | 512 | 47.70% | 50.42% | 46.88% | 2.30 pp | -45 | 51 | -0.88 |
| BTC Hourly | transformer | Transformer | 979 | 454 | 525 | 46.37% | 43.75% | 43.54% | 3.63 pp | -71 | 51 | -1.39 |
| BTC Hourly | rf | RandomForest | 979 | 433 | 546 | 44.23% | 42.08% | 42.92% | 5.77 pp | -113 | 51 | -2.22 |
| BTC Hourly | nn | NN | 979 | 432 | 547 | 44.13% | 41.67% | 42.50% | 5.87 pp | -115 | 51 | -2.25 |
| BTC Hourly | lstm | LSTM | 979 | 417 | 562 | 42.59% | 37.92% | 41.04% | 7.41 pp | -145 | 51 | -2.84 |
| BTC Hourly | xgb | XGBoost | 979 | 405 | 574 | 41.37% | 36.25% | 39.17% | 8.63 pp | -169 | 51 | -3.31 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 802 | 386 | 416 | 48.13% | 46.67% | 47.29% | 1.87 pp | -30 | 46 | -0.65 |
| BTC Daily | nn | NN | 802 | 373 | 429 | 46.51% | 45.00% | 44.79% | 3.49 pp | -56 | 46 | -1.22 |
| BTC Daily | transformer | Transformer | 802 | 371 | 431 | 46.26% | 39.58% | 45.83% | 3.74 pp | -60 | 46 | -1.30 |
| BTC Daily | lstm | LSTM | 802 | 337 | 465 | 42.02% | 34.58% | 40.00% | 7.98 pp | -128 | 46 | -2.78 |
| BTC Daily | rf | RandomForest | 802 | 334 | 468 | 41.65% | 37.08% | 41.04% | 8.35 pp | -134 | 46 | -2.91 |
| BTC Daily | xgb | XGBoost | 812 | 316 | 496 | 38.92% | 35.00% | 35.62% | 11.08 pp | -180 | 46 | -3.91 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 574 | 280 | 294 | 48.78% | 47.92% | 48.12% | 1.22 pp | -14 | 54 | -0.26 |
| BTC Market Hours | nn | NN | 574 | 274 | 300 | 47.74% | 52.08% | 49.38% | 2.26 pp | -26 | 54 | -0.48 |
| BTC Market Hours | transformer | Transformer | 574 | 270 | 304 | 47.04% | 46.67% | 46.88% | 2.96 pp | -34 | 54 | -0.63 |
| BTC Market Hours | lstm | LSTM | 574 | 249 | 325 | 43.38% | 43.33% | 43.75% | 6.62 pp | -76 | 54 | -1.41 |
| BTC Market Hours | rf | RandomForest | 574 | 248 | 326 | 43.21% | 45.42% | 43.75% | 6.79 pp | -78 | 54 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 574 | 248 | 326 | 43.21% | 47.08% | 43.54% | 6.79 pp | -78 | 54 | -1.44 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 628 | 293 | 335 | 46.66% | 47.50% | 47.92% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 628 | 293 | 335 | 46.66% | 49.17% | 47.29% | 3.34 pp | -42 | 53 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 628 | 292 | 336 | 46.50% | 48.33% | 46.88% | 3.50 pp | -44 | 53 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 628 | 262 | 366 | 41.72% | 43.33% | 40.62% | 8.28 pp | -104 | 53 | -1.96 |
| BTC Market Hours Daily | xgb | XGBoost | 628 | 260 | 368 | 41.40% | 44.58% | 41.04% | 8.60 pp | -108 | 53 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 628 | 256 | 372 | 40.76% | 41.67% | 40.00% | 9.24 pp | -116 | 53 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Hourly | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 216 | 103 | 113 | 47.69% | 47.69% | 47.69% | 2.31 pp | -10 | 14 | -0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 216 | 101 | 115 | 46.76% | 46.76% | 46.76% | 3.24 pp | -14 | 14 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 216 | 99 | 117 | 45.83% | 45.83% | 45.83% | 4.17 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 216 | 98 | 118 | 45.37% | 45.37% | 45.37% | 4.63 pp | -20 | 14 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 216 | 94 | 122 | 43.52% | 43.52% | 43.52% | 6.48 pp | -28 | 14 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 216 | 93 | 123 | 43.06% | 43.06% | 43.06% | 6.94 pp | -30 | 14 | -2.14 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 72 | 33 | 39 | 45.83% | 45.83% | 45.83% | 4.17 pp | -6 | 6 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 6 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | rf | RandomForest | 73 | 31 | 42 | 42.47% | 42.47% | 42.47% | 7.53 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 73 | 30 | 43 | 41.10% | 41.10% | 41.10% | 8.90 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 73 | 28 | 45 | 38.36% | 38.36% | 38.36% | 11.64 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 73 | 27 | 46 | 36.99% | 36.99% | 36.99% | 13.01 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
