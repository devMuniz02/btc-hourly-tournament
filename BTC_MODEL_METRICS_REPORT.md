# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T00:11:02.641667+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1331 | 1043 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1207 | 842 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 23:00:00+00:00 | 965 | 604 | 360 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 23:00:00+00:00 | 967 | 658 | 307 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 604 | 290 | 314 | 48.01% | 46.25% | 47.29% | 1.99 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 604 | 287 | 317 | 47.52% | 50.42% | 48.96% | 2.48 pp | -30 | 56 | -0.54 |
| BTC Market Hours | transformer | Transformer | 604 | 282 | 322 | 46.69% | 45.83% | 46.04% | 3.31 pp | -40 | 56 | -0.71 |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 658 | 308 | 350 | 46.81% | 48.75% | 47.08% | 3.19 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | nn | NN | 658 | 308 | 350 | 46.81% | 48.33% | 47.71% | 3.19 pp | -42 | 56 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 832 | 397 | 435 | 47.72% | 44.58% | 46.04% | 2.28 pp | -38 | 47 | -0.81 |
| BTC Market Hours Daily | transformer | Transformer | 658 | 306 | 352 | 46.50% | 47.92% | 47.71% | 3.50 pp | -46 | 56 | -0.82 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1009 | 478 | 531 | 47.37% | 48.33% | 45.62% | 2.63 pp | -53 | 52 | -1.02 |
| BTC Daily | nn | NN | 832 | 387 | 445 | 46.51% | 45.00% | 45.00% | 3.49 pp | -58 | 47 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| BTC Hourly | transformer | Transformer | 1009 | 470 | 539 | 46.58% | 45.83% | 44.58% | 3.42 pp | -69 | 52 | -1.33 |
| BTC Daily | transformer | Transformer | 832 | 384 | 448 | 46.15% | 37.50% | 44.79% | 3.85 pp | -64 | 47 | -1.36 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 604 | 258 | 346 | 42.72% | 42.50% | 42.71% | 7.28 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 604 | 257 | 347 | 42.55% | 42.92% | 42.08% | 7.45 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 604 | 256 | 348 | 42.38% | 45.00% | 43.12% | 7.62 pp | -92 | 56 | -1.64 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 658 | 272 | 386 | 41.34% | 42.50% | 41.25% | 8.66 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 658 | 269 | 389 | 40.88% | 42.92% | 40.42% | 9.12 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 658 | 269 | 389 | 40.88% | 42.92% | 40.62% | 9.12 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1009 | 445 | 564 | 44.10% | 42.50% | 41.25% | 5.90 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1009 | 445 | 564 | 44.10% | 41.67% | 43.33% | 5.90 pp | -119 | 52 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| BTC Daily | lstm | LSTM | 832 | 351 | 481 | 42.19% | 35.42% | 40.00% | 7.81 pp | -130 | 47 | -2.77 |
| BTC Hourly | lstm | LSTM | 1009 | 427 | 582 | 42.32% | 36.25% | 40.00% | 7.68 pp | -155 | 52 | -2.98 |
| BTC Daily | rf | RandomForest | 832 | 345 | 487 | 41.47% | 36.67% | 40.62% | 8.53 pp | -142 | 47 | -3.02 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | xgb | XGBoost | 1009 | 415 | 594 | 41.13% | 35.42% | 38.33% | 8.87 pp | -179 | 52 | -3.44 |
| BTC Daily | xgb | XGBoost | 842 | 332 | 510 | 39.43% | 37.50% | 36.88% | 10.57 pp | -178 | 47 | -3.79 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1009 | 478 | 531 | 47.37% | 48.33% | 45.62% | 2.63 pp | -53 | 52 | -1.02 |
| BTC Hourly | transformer | Transformer | 1009 | 470 | 539 | 46.58% | 45.83% | 44.58% | 3.42 pp | -69 | 52 | -1.33 |
| BTC Hourly | nn | NN | 1009 | 445 | 564 | 44.10% | 42.50% | 41.25% | 5.90 pp | -119 | 52 | -2.29 |
| BTC Hourly | rf | RandomForest | 1009 | 445 | 564 | 44.10% | 41.67% | 43.33% | 5.90 pp | -119 | 52 | -2.29 |
| BTC Hourly | lstm | LSTM | 1009 | 427 | 582 | 42.32% | 36.25% | 40.00% | 7.68 pp | -155 | 52 | -2.98 |
| BTC Hourly | xgb | XGBoost | 1009 | 415 | 594 | 41.13% | 35.42% | 38.33% | 8.87 pp | -179 | 52 | -3.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 832 | 397 | 435 | 47.72% | 44.58% | 46.04% | 2.28 pp | -38 | 47 | -0.81 |
| BTC Daily | nn | NN | 832 | 387 | 445 | 46.51% | 45.00% | 45.00% | 3.49 pp | -58 | 47 | -1.23 |
| BTC Daily | transformer | Transformer | 832 | 384 | 448 | 46.15% | 37.50% | 44.79% | 3.85 pp | -64 | 47 | -1.36 |
| BTC Daily | lstm | LSTM | 832 | 351 | 481 | 42.19% | 35.42% | 40.00% | 7.81 pp | -130 | 47 | -2.77 |
| BTC Daily | rf | RandomForest | 832 | 345 | 487 | 41.47% | 36.67% | 40.62% | 8.53 pp | -142 | 47 | -3.02 |
| BTC Daily | xgb | XGBoost | 842 | 332 | 510 | 39.43% | 37.50% | 36.88% | 10.57 pp | -178 | 47 | -3.79 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 604 | 290 | 314 | 48.01% | 46.25% | 47.29% | 1.99 pp | -24 | 56 | -0.43 |
| BTC Market Hours | nn | NN | 604 | 287 | 317 | 47.52% | 50.42% | 48.96% | 2.48 pp | -30 | 56 | -0.54 |
| BTC Market Hours | transformer | Transformer | 604 | 282 | 322 | 46.69% | 45.83% | 46.04% | 3.31 pp | -40 | 56 | -0.71 |
| BTC Market Hours | lstm | LSTM | 604 | 258 | 346 | 42.72% | 42.50% | 42.71% | 7.28 pp | -88 | 56 | -1.57 |
| BTC Market Hours | rf | RandomForest | 604 | 257 | 347 | 42.55% | 42.92% | 42.08% | 7.45 pp | -90 | 56 | -1.61 |
| BTC Market Hours | xgb | XGBoost | 604 | 256 | 348 | 42.38% | 45.00% | 43.12% | 7.62 pp | -92 | 56 | -1.64 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 658 | 308 | 350 | 46.81% | 48.75% | 47.08% | 3.19 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | nn | NN | 658 | 308 | 350 | 46.81% | 48.33% | 47.71% | 3.19 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 658 | 306 | 352 | 46.50% | 47.92% | 47.71% | 3.50 pp | -46 | 56 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 658 | 272 | 386 | 41.34% | 42.50% | 41.25% | 8.66 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 658 | 269 | 389 | 40.88% | 42.92% | 40.42% | 9.12 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 658 | 269 | 389 | 40.88% | 42.92% | 40.62% | 9.12 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
