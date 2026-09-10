# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T18:26:50.185565+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1328 | 1040 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1204 | 839 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 17:00:00+00:00 | 956 | 601 | 354 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 17:00:00+00:00 | 958 | 655 | 301 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 241 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 241 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 86 | 155 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 18:00:00+00:00 | 241 | 86 | 155 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 601 | 289 | 312 | 48.09% | 46.25% | 47.08% | 1.91 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 601 | 287 | 314 | 47.75% | 51.25% | 49.17% | 2.25 pp | -27 | 56 | -0.48 |
| BTC Market Hours | transformer | Transformer | 601 | 282 | 319 | 46.92% | 45.83% | 46.04% | 3.08 pp | -37 | 56 | -0.66 |
| BTC Market Hours Daily | nn | NN | 655 | 307 | 348 | 46.87% | 48.33% | 47.71% | 3.13 pp | -41 | 56 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 655 | 306 | 349 | 46.72% | 48.33% | 47.08% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 655 | 306 | 349 | 46.72% | 48.33% | 48.12% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 829 | 395 | 434 | 47.65% | 45.00% | 46.04% | 2.35 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1006 | 477 | 529 | 47.42% | 48.33% | 45.83% | 2.58 pp | -52 | 52 | -1.00 |
| BTC Daily | nn | NN | 829 | 386 | 443 | 46.56% | 45.42% | 45.42% | 3.44 pp | -57 | 47 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| BTC Daily | transformer | Transformer | 829 | 383 | 446 | 46.20% | 38.33% | 45.00% | 3.80 pp | -63 | 47 | -1.34 |
| BTC Hourly | transformer | Transformer | 1006 | 468 | 538 | 46.52% | 45.83% | 44.58% | 3.48 pp | -70 | 52 | -1.35 |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 601 | 257 | 344 | 42.76% | 42.08% | 42.71% | 7.24 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 601 | 256 | 345 | 42.60% | 42.50% | 42.08% | 7.40 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 601 | 255 | 346 | 42.43% | 45.00% | 43.33% | 7.57 pp | -91 | 56 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 655 | 271 | 384 | 41.37% | 42.08% | 41.25% | 8.63 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 655 | 268 | 387 | 40.92% | 42.50% | 40.42% | 9.08 pp | -119 | 56 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 655 | 268 | 387 | 40.92% | 42.50% | 40.62% | 9.08 pp | -119 | 56 | -2.12 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| BTC Hourly | nn | NN | 1006 | 443 | 563 | 44.04% | 42.08% | 41.25% | 5.96 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1006 | 443 | 563 | 44.04% | 41.67% | 43.12% | 5.96 pp | -120 | 52 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| BTC Daily | lstm | LSTM | 829 | 349 | 480 | 42.10% | 35.42% | 40.00% | 7.90 pp | -131 | 47 | -2.79 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| BTC Hourly | lstm | LSTM | 1006 | 425 | 581 | 42.25% | 36.25% | 40.00% | 7.75 pp | -156 | 52 | -3.00 |
| BTC Daily | rf | RandomForest | 829 | 343 | 486 | 41.38% | 37.08% | 40.42% | 8.62 pp | -143 | 47 | -3.04 |
| Consolidated Hourly | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| BTC Hourly | xgb | XGBoost | 1006 | 414 | 592 | 41.15% | 35.83% | 38.54% | 8.85 pp | -178 | 52 | -3.42 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 839 | 331 | 508 | 39.45% | 37.50% | 36.88% | 10.55 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1006 | 477 | 529 | 47.42% | 48.33% | 45.83% | 2.58 pp | -52 | 52 | -1.00 |
| BTC Hourly | transformer | Transformer | 1006 | 468 | 538 | 46.52% | 45.83% | 44.58% | 3.48 pp | -70 | 52 | -1.35 |
| BTC Hourly | nn | NN | 1006 | 443 | 563 | 44.04% | 42.08% | 41.25% | 5.96 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1006 | 443 | 563 | 44.04% | 41.67% | 43.12% | 5.96 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 1006 | 425 | 581 | 42.25% | 36.25% | 40.00% | 7.75 pp | -156 | 52 | -3.00 |
| BTC Hourly | xgb | XGBoost | 1006 | 414 | 592 | 41.15% | 35.83% | 38.54% | 8.85 pp | -178 | 52 | -3.42 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 829 | 395 | 434 | 47.65% | 45.00% | 46.04% | 2.35 pp | -39 | 47 | -0.83 |
| BTC Daily | nn | NN | 829 | 386 | 443 | 46.56% | 45.42% | 45.42% | 3.44 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 829 | 383 | 446 | 46.20% | 38.33% | 45.00% | 3.80 pp | -63 | 47 | -1.34 |
| BTC Daily | lstm | LSTM | 829 | 349 | 480 | 42.10% | 35.42% | 40.00% | 7.90 pp | -131 | 47 | -2.79 |
| BTC Daily | rf | RandomForest | 829 | 343 | 486 | 41.38% | 37.08% | 40.42% | 8.62 pp | -143 | 47 | -3.04 |
| BTC Daily | xgb | XGBoost | 839 | 331 | 508 | 39.45% | 37.50% | 36.88% | 10.55 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 601 | 289 | 312 | 48.09% | 46.25% | 47.08% | 1.91 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 601 | 287 | 314 | 47.75% | 51.25% | 49.17% | 2.25 pp | -27 | 56 | -0.48 |
| BTC Market Hours | transformer | Transformer | 601 | 282 | 319 | 46.92% | 45.83% | 46.04% | 3.08 pp | -37 | 56 | -0.66 |
| BTC Market Hours | lstm | LSTM | 601 | 257 | 344 | 42.76% | 42.08% | 42.71% | 7.24 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 601 | 256 | 345 | 42.60% | 42.50% | 42.08% | 7.40 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 601 | 255 | 346 | 42.43% | 45.00% | 43.33% | 7.57 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 655 | 307 | 348 | 46.87% | 48.33% | 47.71% | 3.13 pp | -41 | 56 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 655 | 306 | 349 | 46.72% | 48.33% | 47.08% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | transformer | Transformer | 655 | 306 | 349 | 46.72% | 48.33% | 48.12% | 3.28 pp | -43 | 56 | -0.77 |
| BTC Market Hours Daily | rf | RandomForest | 655 | 271 | 384 | 41.37% | 42.08% | 41.25% | 8.63 pp | -113 | 56 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 655 | 268 | 387 | 40.92% | 42.50% | 40.42% | 9.08 pp | -119 | 56 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 655 | 268 | 387 | 40.92% | 42.50% | 40.62% | 9.08 pp | -119 | 56 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 115 | 126 | 47.72% | 47.92% | 47.72% | 2.28 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 111 | 130 | 46.06% | 45.83% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 107 | 134 | 44.40% | 44.17% | 44.40% | 5.60 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 101 | 140 | 41.91% | 41.67% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 97 | 144 | 40.25% | 40.42% | 40.25% | 9.75 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
