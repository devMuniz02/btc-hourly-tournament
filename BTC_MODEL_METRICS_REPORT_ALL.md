# BTC Model Metrics Report - All Rows

Generated at: 2026-09-11T17:00:21.094066+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1343 | 1055 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1218 | 853 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 981 | 615 | 365 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 983 | 669 | 312 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 615 | 296 | 319 | 48.13% | 46.67% | 47.29% | 1.87 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 615 | 294 | 321 | 47.80% | 50.42% | 49.58% | 2.20 pp | -27 | 57 | -0.47 |
| BTC Market Hours Daily | nn | NN | 669 | 316 | 353 | 47.23% | 50.42% | 48.75% | 2.77 pp | -37 | 57 | -0.65 |
| BTC Market Hours | transformer | Transformer | 615 | 287 | 328 | 46.67% | 46.25% | 45.62% | 3.33 pp | -41 | 57 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 669 | 313 | 356 | 46.79% | 49.17% | 47.29% | 3.21 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 669 | 310 | 359 | 46.34% | 47.92% | 47.92% | 3.66 pp | -49 | 57 | -0.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 843 | 400 | 443 | 47.45% | 43.75% | 45.83% | 2.55 pp | -43 | 48 | -0.90 |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1021 | 483 | 538 | 47.31% | 47.08% | 46.25% | 2.69 pp | -55 | 53 | -1.04 |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| BTC Daily | nn | NN | 843 | 393 | 450 | 46.62% | 45.83% | 45.21% | 3.38 pp | -57 | 48 | -1.19 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| BTC Hourly | transformer | Transformer | 1021 | 479 | 542 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 53 | -1.19 |
| BTC Daily | transformer | Transformer | 843 | 388 | 455 | 46.03% | 37.92% | 44.17% | 3.97 pp | -67 | 48 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| BTC Market Hours | lstm | LSTM | 615 | 264 | 351 | 42.93% | 43.33% | 43.12% | 7.07 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 615 | 263 | 352 | 42.76% | 43.33% | 41.67% | 7.24 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 615 | 262 | 353 | 42.60% | 46.67% | 43.12% | 7.40 pp | -91 | 57 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 669 | 278 | 391 | 41.55% | 43.33% | 41.88% | 8.45 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 669 | 275 | 394 | 41.11% | 44.17% | 41.25% | 8.89 pp | -119 | 57 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 669 | 274 | 395 | 40.96% | 43.75% | 41.04% | 9.04 pp | -121 | 57 | -2.12 |
| BTC Hourly | nn | NN | 1021 | 449 | 572 | 43.98% | 40.83% | 40.83% | 6.02 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1021 | 447 | 574 | 43.78% | 40.42% | 42.29% | 6.22 pp | -127 | 53 | -2.40 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Daily | lstm | LSTM | 843 | 356 | 487 | 42.23% | 35.83% | 39.79% | 7.77 pp | -131 | 48 | -2.73 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| BTC Daily | rf | RandomForest | 843 | 350 | 493 | 41.52% | 37.50% | 40.83% | 8.48 pp | -143 | 48 | -2.98 |
| BTC Hourly | lstm | LSTM | 1021 | 429 | 592 | 42.02% | 34.58% | 39.38% | 7.98 pp | -163 | 53 | -3.08 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | xgb | XGBoost | 1021 | 419 | 602 | 41.04% | 34.58% | 37.71% | 8.96 pp | -183 | 53 | -3.45 |
| BTC Daily | xgb | XGBoost | 853 | 336 | 517 | 39.39% | 37.92% | 36.25% | 10.61 pp | -181 | 48 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1021 | 483 | 538 | 47.31% | 47.08% | 46.25% | 2.69 pp | -55 | 53 | -1.04 |
| BTC Hourly | transformer | Transformer | 1021 | 479 | 542 | 46.91% | 47.08% | 45.00% | 3.09 pp | -63 | 53 | -1.19 |
| BTC Hourly | nn | NN | 1021 | 449 | 572 | 43.98% | 40.83% | 40.83% | 6.02 pp | -123 | 53 | -2.32 |
| BTC Hourly | rf | RandomForest | 1021 | 447 | 574 | 43.78% | 40.42% | 42.29% | 6.22 pp | -127 | 53 | -2.40 |
| BTC Hourly | lstm | LSTM | 1021 | 429 | 592 | 42.02% | 34.58% | 39.38% | 7.98 pp | -163 | 53 | -3.08 |
| BTC Hourly | xgb | XGBoost | 1021 | 419 | 602 | 41.04% | 34.58% | 37.71% | 8.96 pp | -183 | 53 | -3.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 843 | 400 | 443 | 47.45% | 43.75% | 45.83% | 2.55 pp | -43 | 48 | -0.90 |
| BTC Daily | nn | NN | 843 | 393 | 450 | 46.62% | 45.83% | 45.21% | 3.38 pp | -57 | 48 | -1.19 |
| BTC Daily | transformer | Transformer | 843 | 388 | 455 | 46.03% | 37.92% | 44.17% | 3.97 pp | -67 | 48 | -1.40 |
| BTC Daily | lstm | LSTM | 843 | 356 | 487 | 42.23% | 35.83% | 39.79% | 7.77 pp | -131 | 48 | -2.73 |
| BTC Daily | rf | RandomForest | 843 | 350 | 493 | 41.52% | 37.50% | 40.83% | 8.48 pp | -143 | 48 | -2.98 |
| BTC Daily | xgb | XGBoost | 853 | 336 | 517 | 39.39% | 37.92% | 36.25% | 10.61 pp | -181 | 48 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 615 | 296 | 319 | 48.13% | 46.67% | 47.29% | 1.87 pp | -23 | 57 | -0.40 |
| BTC Market Hours | nn | NN | 615 | 294 | 321 | 47.80% | 50.42% | 49.58% | 2.20 pp | -27 | 57 | -0.47 |
| BTC Market Hours | transformer | Transformer | 615 | 287 | 328 | 46.67% | 46.25% | 45.62% | 3.33 pp | -41 | 57 | -0.72 |
| BTC Market Hours | lstm | LSTM | 615 | 264 | 351 | 42.93% | 43.33% | 43.12% | 7.07 pp | -87 | 57 | -1.53 |
| BTC Market Hours | rf | RandomForest | 615 | 263 | 352 | 42.76% | 43.33% | 41.67% | 7.24 pp | -89 | 57 | -1.56 |
| BTC Market Hours | xgb | XGBoost | 615 | 262 | 353 | 42.60% | 46.67% | 43.12% | 7.40 pp | -91 | 57 | -1.60 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 669 | 316 | 353 | 47.23% | 50.42% | 48.75% | 2.77 pp | -37 | 57 | -0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 669 | 313 | 356 | 46.79% | 49.17% | 47.29% | 3.21 pp | -43 | 57 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 669 | 310 | 359 | 46.34% | 47.92% | 47.92% | 3.66 pp | -49 | 57 | -0.86 |
| BTC Market Hours Daily | rf | RandomForest | 669 | 278 | 391 | 41.55% | 43.33% | 41.88% | 8.45 pp | -113 | 57 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 669 | 275 | 394 | 41.11% | 44.17% | 41.25% | 8.89 pp | -119 | 57 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 669 | 274 | 395 | 40.96% | 43.75% | 41.04% | 9.04 pp | -121 | 57 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
