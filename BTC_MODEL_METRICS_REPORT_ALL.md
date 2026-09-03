# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T05:11:49.558683+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1206 | 918 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1082 | 717 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 737 | 479 | 257 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 739 | 533 | 204 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T22:00:00+00:00 | 129 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T22:00:00+00:00 | 129 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T22:00:00+00:00 | 129 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T22:00:00+00:00 | 130 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 66 | 63 | 51.16% | 51.16% | 51.16% | 1.16 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 66 | 63 | 51.16% | 51.16% | 51.16% | 1.16 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 707 | 346 | 361 | 48.94% | 47.50% | 48.96% | 1.06 pp | -15 | 42 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 479 | 230 | 249 | 48.02% | 43.33% | 48.02% | 1.98 pp | -19 | 46 | -0.41 |
| Consolidated Hourly | xgb | XGBoost | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 479 | 226 | 253 | 47.18% | 48.33% | 47.18% | 2.82 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 479 | 224 | 255 | 46.76% | 41.67% | 46.76% | 3.24 pp | -31 | 46 | -0.67 |
| BTC Daily | transformer | Transformer | 707 | 339 | 368 | 47.95% | 47.50% | 50.21% | 2.05 pp | -29 | 42 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 533 | 247 | 286 | 46.34% | 49.17% | 47.29% | 3.66 pp | -39 | 46 | -0.85 |
| Consolidated Hourly | lstm | LSTM | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | nn | NN | 533 | 244 | 289 | 45.78% | 43.33% | 46.46% | 4.22 pp | -45 | 46 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 884 | 419 | 465 | 47.40% | 48.75% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 884 | 419 | 465 | 47.40% | 48.33% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 533 | 243 | 290 | 45.59% | 47.08% | 46.46% | 4.41 pp | -47 | 46 | -1.02 |
| BTC Daily | nn | NN | 707 | 328 | 379 | 46.39% | 43.75% | 48.12% | 3.61 pp | -51 | 42 | -1.21 |
| BTC Market Hours | lstm | LSTM | 479 | 207 | 272 | 43.22% | 41.67% | 43.22% | 6.78 pp | -65 | 46 | -1.41 |
| Consolidated Hourly | nn | NN | 129 | 57 | 72 | 44.19% | 44.19% | 44.19% | 5.81 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 57 | 72 | 44.19% | 44.19% | 44.19% | 5.81 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| BTC Market Hours | rf | RandomForest | 479 | 204 | 275 | 42.59% | 41.67% | 42.59% | 7.41 pp | -71 | 46 | -1.54 |
| Consolidated Hourly | transformer | Transformer | 129 | 56 | 73 | 43.41% | 43.41% | 43.41% | 6.59 pp | -17 | 10 | -1.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 56 | 73 | 43.41% | 43.41% | 43.41% | 6.59 pp | -17 | 10 | -1.70 |
| BTC Market Hours | xgb | XGBoost | 479 | 195 | 284 | 40.71% | 39.58% | 40.71% | 9.29 pp | -89 | 46 | -1.93 |
| BTC Hourly | nn | NN | 884 | 396 | 488 | 44.80% | 45.42% | 43.33% | 5.20 pp | -92 | 47 | -1.96 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Hourly | rf | RandomForest | 884 | 394 | 490 | 44.57% | 44.17% | 44.17% | 5.43 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 533 | 219 | 314 | 41.09% | 40.83% | 41.25% | 8.91 pp | -95 | 46 | -2.07 |
| BTC Daily | lstm | LSTM | 707 | 307 | 400 | 43.42% | 39.17% | 42.50% | 6.58 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 707 | 304 | 403 | 43.00% | 42.08% | 43.54% | 7.00 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 533 | 212 | 321 | 39.77% | 38.75% | 39.17% | 10.23 pp | -109 | 46 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 533 | 211 | 322 | 39.59% | 36.67% | 40.21% | 10.41 pp | -111 | 46 | -2.41 |
| BTC Hourly | lstm | LSTM | 884 | 378 | 506 | 42.76% | 38.33% | 42.29% | 7.24 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 884 | 374 | 510 | 42.31% | 42.08% | 42.29% | 7.69 pp | -136 | 47 | -2.89 |
| BTC Daily | xgb | XGBoost | 717 | 282 | 435 | 39.33% | 34.17% | 39.38% | 10.67 pp | -153 | 42 | -3.64 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 884 | 419 | 465 | 47.40% | 48.75% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | transformer | Transformer | 884 | 419 | 465 | 47.40% | 48.33% | 47.71% | 2.60 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 884 | 396 | 488 | 44.80% | 45.42% | 43.33% | 5.20 pp | -92 | 47 | -1.96 |
| BTC Hourly | rf | RandomForest | 884 | 394 | 490 | 44.57% | 44.17% | 44.17% | 5.43 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 884 | 378 | 506 | 42.76% | 38.33% | 42.29% | 7.24 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 884 | 374 | 510 | 42.31% | 42.08% | 42.29% | 7.69 pp | -136 | 47 | -2.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 707 | 346 | 361 | 48.94% | 47.50% | 48.96% | 1.06 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 707 | 339 | 368 | 47.95% | 47.50% | 50.21% | 2.05 pp | -29 | 42 | -0.69 |
| BTC Daily | nn | NN | 707 | 328 | 379 | 46.39% | 43.75% | 48.12% | 3.61 pp | -51 | 42 | -1.21 |
| BTC Daily | lstm | LSTM | 707 | 307 | 400 | 43.42% | 39.17% | 42.50% | 6.58 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 707 | 304 | 403 | 43.00% | 42.08% | 43.54% | 7.00 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 717 | 282 | 435 | 39.33% | 34.17% | 39.38% | 10.67 pp | -153 | 42 | -3.64 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 479 | 230 | 249 | 48.02% | 43.33% | 48.02% | 1.98 pp | -19 | 46 | -0.41 |
| BTC Market Hours | nn | NN | 479 | 226 | 253 | 47.18% | 48.33% | 47.18% | 2.82 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 479 | 224 | 255 | 46.76% | 41.67% | 46.76% | 3.24 pp | -31 | 46 | -0.67 |
| BTC Market Hours | lstm | LSTM | 479 | 207 | 272 | 43.22% | 41.67% | 43.22% | 6.78 pp | -65 | 46 | -1.41 |
| BTC Market Hours | rf | RandomForest | 479 | 204 | 275 | 42.59% | 41.67% | 42.59% | 7.41 pp | -71 | 46 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 479 | 195 | 284 | 40.71% | 39.58% | 40.71% | 9.29 pp | -89 | 46 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 533 | 247 | 286 | 46.34% | 49.17% | 47.29% | 3.66 pp | -39 | 46 | -0.85 |
| BTC Market Hours Daily | nn | NN | 533 | 244 | 289 | 45.78% | 43.33% | 46.46% | 4.22 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 533 | 243 | 290 | 45.59% | 47.08% | 46.46% | 4.41 pp | -47 | 46 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 533 | 219 | 314 | 41.09% | 40.83% | 41.25% | 8.91 pp | -95 | 46 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 533 | 212 | 321 | 39.77% | 38.75% | 39.17% | 10.23 pp | -109 | 46 | -2.37 |
| BTC Market Hours Daily | lstm | LSTM | 533 | 211 | 322 | 39.59% | 36.67% | 40.21% | 10.41 pp | -111 | 46 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 129 | 66 | 63 | 51.16% | 51.16% | 51.16% | 1.16 pp | 3 | 10 | 0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | xgb | XGBoost | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | nn | NN | 129 | 57 | 72 | 44.19% | 44.19% | 44.19% | 5.81 pp | -15 | 10 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 129 | 56 | 73 | 43.41% | 43.41% | 43.41% | 6.59 pp | -17 | 10 | -1.70 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 129 | 66 | 63 | 51.16% | 51.16% | 51.16% | 1.16 pp | 3 | 10 | 0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 129 | 63 | 66 | 48.84% | 48.84% | 48.84% | 1.16 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 129 | 62 | 67 | 48.06% | 48.06% | 48.06% | 1.94 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 129 | 60 | 69 | 46.51% | 46.51% | 46.51% | 3.49 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 129 | 57 | 72 | 44.19% | 44.19% | 44.19% | 5.81 pp | -15 | 10 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 129 | 56 | 73 | 43.41% | 43.41% | 43.41% | 6.59 pp | -17 | 10 | -1.70 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 25 | 12 | 13 | 48.00% | 48.00% | 48.00% | 2.00 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 25 | 8 | 17 | 32.00% | 32.00% | 32.00% | 18.00 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
