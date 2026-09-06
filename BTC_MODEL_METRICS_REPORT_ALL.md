# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T20:26:26.109254+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1265 | 977 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1141 | 776 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 19:00:00+00:00 | 843 | 538 | 304 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 19:00:00+00:00 | 845 | 592 | 251 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 183 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T13:00:00+00:00 | 184 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 538 | 262 | 276 | 48.70% | 46.67% | 48.33% | 1.30 pp | -14 | 51 | -0.27 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 538 | 258 | 280 | 47.96% | 49.58% | 48.54% | 2.04 pp | -22 | 51 | -0.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 766 | 372 | 394 | 48.56% | 47.92% | 48.75% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Market Hours | nn | NN | 538 | 255 | 283 | 47.40% | 50.83% | 48.96% | 2.60 pp | -28 | 51 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 592 | 281 | 311 | 47.47% | 50.83% | 48.75% | 2.53 pp | -30 | 51 | -0.59 |
| BTC Market Hours Daily | nn | NN | 592 | 276 | 316 | 46.62% | 46.25% | 48.12% | 3.38 pp | -40 | 51 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 592 | 275 | 317 | 46.45% | 50.83% | 47.29% | 3.55 pp | -42 | 51 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 943 | 450 | 493 | 47.72% | 50.00% | 46.88% | 2.28 pp | -43 | 49 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 766 | 360 | 406 | 47.00% | 42.50% | 47.50% | 3.00 pp | -46 | 45 | -1.02 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 943 | 442 | 501 | 46.87% | 46.67% | 45.00% | 3.13 pp | -59 | 49 | -1.20 |
| BTC Daily | nn | NN | 766 | 355 | 411 | 46.34% | 45.00% | 46.25% | 3.66 pp | -56 | 45 | -1.24 |
| BTC Market Hours | rf | RandomForest | 538 | 233 | 305 | 43.31% | 45.42% | 43.54% | 6.69 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 538 | 232 | 306 | 43.12% | 42.08% | 43.96% | 6.88 pp | -74 | 51 | -1.45 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 538 | 223 | 315 | 41.45% | 43.33% | 41.88% | 8.55 pp | -92 | 51 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 592 | 247 | 345 | 41.72% | 45.00% | 41.46% | 8.28 pp | -98 | 51 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| BTC Hourly | nn | NN | 943 | 419 | 524 | 44.43% | 42.92% | 42.71% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 943 | 419 | 524 | 44.43% | 44.58% | 43.75% | 5.57 pp | -105 | 49 | -2.14 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 592 | 239 | 353 | 40.37% | 38.75% | 40.00% | 9.63 pp | -114 | 51 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 592 | 236 | 356 | 39.86% | 41.25% | 38.96% | 10.14 pp | -120 | 51 | -2.35 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 766 | 324 | 442 | 42.30% | 35.83% | 40.42% | 7.70 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 766 | 321 | 445 | 41.91% | 38.33% | 42.29% | 8.09 pp | -124 | 45 | -2.76 |
| BTC Hourly | lstm | LSTM | 943 | 402 | 541 | 42.63% | 36.25% | 41.88% | 7.37 pp | -139 | 49 | -2.84 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | xgb | XGBoost | 943 | 396 | 547 | 41.99% | 40.83% | 40.62% | 8.01 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 5 | -3.20 |
| BTC Daily | xgb | XGBoost | 776 | 305 | 471 | 39.30% | 35.83% | 36.67% | 10.70 pp | -166 | 45 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 943 | 450 | 493 | 47.72% | 50.00% | 46.88% | 2.28 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 943 | 442 | 501 | 46.87% | 46.67% | 45.00% | 3.13 pp | -59 | 49 | -1.20 |
| BTC Hourly | nn | NN | 943 | 419 | 524 | 44.43% | 42.92% | 42.71% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | rf | RandomForest | 943 | 419 | 524 | 44.43% | 44.58% | 43.75% | 5.57 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 943 | 402 | 541 | 42.63% | 36.25% | 41.88% | 7.37 pp | -139 | 49 | -2.84 |
| BTC Hourly | xgb | XGBoost | 943 | 396 | 547 | 41.99% | 40.83% | 40.62% | 8.01 pp | -151 | 49 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 766 | 372 | 394 | 48.56% | 47.92% | 48.75% | 1.44 pp | -22 | 45 | -0.49 |
| BTC Daily | transformer | Transformer | 766 | 360 | 406 | 47.00% | 42.50% | 47.50% | 3.00 pp | -46 | 45 | -1.02 |
| BTC Daily | nn | NN | 766 | 355 | 411 | 46.34% | 45.00% | 46.25% | 3.66 pp | -56 | 45 | -1.24 |
| BTC Daily | lstm | LSTM | 766 | 324 | 442 | 42.30% | 35.83% | 40.42% | 7.70 pp | -118 | 45 | -2.62 |
| BTC Daily | rf | RandomForest | 766 | 321 | 445 | 41.91% | 38.33% | 42.29% | 8.09 pp | -124 | 45 | -2.76 |
| BTC Daily | xgb | XGBoost | 776 | 305 | 471 | 39.30% | 35.83% | 36.67% | 10.70 pp | -166 | 45 | -3.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 538 | 262 | 276 | 48.70% | 46.67% | 48.33% | 1.30 pp | -14 | 51 | -0.27 |
| BTC Market Hours | transformer | Transformer | 538 | 258 | 280 | 47.96% | 49.58% | 48.54% | 2.04 pp | -22 | 51 | -0.43 |
| BTC Market Hours | nn | NN | 538 | 255 | 283 | 47.40% | 50.83% | 48.96% | 2.60 pp | -28 | 51 | -0.55 |
| BTC Market Hours | rf | RandomForest | 538 | 233 | 305 | 43.31% | 45.42% | 43.54% | 6.69 pp | -72 | 51 | -1.41 |
| BTC Market Hours | lstm | LSTM | 538 | 232 | 306 | 43.12% | 42.08% | 43.96% | 6.88 pp | -74 | 51 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 538 | 223 | 315 | 41.45% | 43.33% | 41.88% | 8.55 pp | -92 | 51 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 592 | 281 | 311 | 47.47% | 50.83% | 48.75% | 2.53 pp | -30 | 51 | -0.59 |
| BTC Market Hours Daily | nn | NN | 592 | 276 | 316 | 46.62% | 46.25% | 48.12% | 3.38 pp | -40 | 51 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 592 | 275 | 317 | 46.45% | 50.83% | 47.29% | 3.55 pp | -42 | 51 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 592 | 247 | 345 | 41.72% | 45.00% | 41.46% | 8.28 pp | -98 | 51 | -1.92 |
| BTC Market Hours Daily | lstm | LSTM | 592 | 239 | 353 | 40.37% | 38.75% | 40.00% | 9.63 pp | -114 | 51 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 592 | 236 | 356 | 39.86% | 41.25% | 38.96% | 10.14 pp | -120 | 51 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 85 | 98 | 46.45% | 46.45% | 46.45% | 3.55 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 82 | 101 | 44.81% | 44.81% | 44.81% | 5.19 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 79 | 104 | 43.17% | 43.17% | 43.17% | 6.83 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 56 | 25 | 31 | 44.64% | 44.64% | 44.64% | 5.36 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 56 | 22 | 34 | 39.29% | 39.29% | 39.29% | 10.71 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
