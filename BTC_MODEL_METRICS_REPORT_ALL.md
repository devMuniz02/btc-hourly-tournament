# BTC Model Metrics Report - All Rows

Generated at: 2026-09-04T07:22:02.487471+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1224 | 936 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1099 | 734 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 767 | 496 | 270 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 769 | 550 | 217 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 496 | 240 | 256 | 48.39% | 44.58% | 48.33% | 1.61 pp | -16 | 48 | -0.33 |
| BTC Market Hours | nn | NN | 496 | 234 | 262 | 47.18% | 50.00% | 47.71% | 2.82 pp | -28 | 48 | -0.58 |
| BTC Market Hours | transformer | Transformer | 496 | 233 | 263 | 46.98% | 44.17% | 47.71% | 3.02 pp | -30 | 48 | -0.62 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 724 | 348 | 376 | 48.07% | 45.83% | 47.50% | 1.93 pp | -28 | 43 | -0.65 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 724 | 345 | 379 | 47.65% | 46.25% | 49.79% | 2.35 pp | -34 | 43 | -0.79 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 902 | 432 | 470 | 47.89% | 51.25% | 48.33% | 2.11 pp | -38 | 48 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 550 | 256 | 294 | 46.55% | 49.17% | 47.50% | 3.45 pp | -38 | 47 | -0.81 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | nn | NN | 550 | 254 | 296 | 46.18% | 45.00% | 47.50% | 3.82 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 550 | 253 | 297 | 46.00% | 48.75% | 46.88% | 4.00 pp | -44 | 47 | -0.94 |
| BTC Hourly | transformer | Transformer | 902 | 427 | 475 | 47.34% | 47.92% | 46.88% | 2.66 pp | -48 | 48 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| BTC Daily | nn | NN | 724 | 335 | 389 | 46.27% | 44.58% | 47.50% | 3.73 pp | -54 | 43 | -1.26 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 496 | 214 | 282 | 43.15% | 40.83% | 43.12% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 496 | 214 | 282 | 43.15% | 43.75% | 43.54% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 48 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | rf | RandomForest | 550 | 229 | 321 | 41.64% | 42.50% | 41.04% | 8.36 pp | -92 | 47 | -1.96 |
| BTC Hourly | nn | NN | 902 | 402 | 500 | 44.57% | 44.58% | 42.29% | 5.43 pp | -98 | 48 | -2.04 |
| BTC Hourly | rf | RandomForest | 902 | 401 | 501 | 44.46% | 44.17% | 44.17% | 5.54 pp | -100 | 48 | -2.08 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 550 | 222 | 328 | 40.36% | 38.75% | 40.62% | 9.64 pp | -106 | 47 | -2.26 |
| BTC Daily | lstm | LSTM | 724 | 312 | 412 | 43.09% | 37.08% | 41.67% | 6.91 pp | -100 | 43 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 550 | 220 | 330 | 40.00% | 41.25% | 39.38% | 10.00 pp | -110 | 47 | -2.34 |
| BTC Daily | rf | RandomForest | 724 | 309 | 415 | 42.68% | 40.83% | 43.54% | 7.32 pp | -106 | 43 | -2.47 |
| BTC Hourly | lstm | LSTM | 902 | 387 | 515 | 42.90% | 40.00% | 42.29% | 7.10 pp | -128 | 48 | -2.67 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | xgb | XGBoost | 902 | 379 | 523 | 42.02% | 41.67% | 41.46% | 7.98 pp | -144 | 48 | -3.00 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 734 | 291 | 443 | 39.65% | 36.67% | 38.75% | 10.35 pp | -152 | 43 | -3.53 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 902 | 432 | 470 | 47.89% | 51.25% | 48.33% | 2.11 pp | -38 | 48 | -0.79 |
| BTC Hourly | transformer | Transformer | 902 | 427 | 475 | 47.34% | 47.92% | 46.88% | 2.66 pp | -48 | 48 | -1.00 |
| BTC Hourly | nn | NN | 902 | 402 | 500 | 44.57% | 44.58% | 42.29% | 5.43 pp | -98 | 48 | -2.04 |
| BTC Hourly | rf | RandomForest | 902 | 401 | 501 | 44.46% | 44.17% | 44.17% | 5.54 pp | -100 | 48 | -2.08 |
| BTC Hourly | lstm | LSTM | 902 | 387 | 515 | 42.90% | 40.00% | 42.29% | 7.10 pp | -128 | 48 | -2.67 |
| BTC Hourly | xgb | XGBoost | 902 | 379 | 523 | 42.02% | 41.67% | 41.46% | 7.98 pp | -144 | 48 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 724 | 348 | 376 | 48.07% | 45.83% | 47.50% | 1.93 pp | -28 | 43 | -0.65 |
| BTC Daily | transformer | Transformer | 724 | 345 | 379 | 47.65% | 46.25% | 49.79% | 2.35 pp | -34 | 43 | -0.79 |
| BTC Daily | nn | NN | 724 | 335 | 389 | 46.27% | 44.58% | 47.50% | 3.73 pp | -54 | 43 | -1.26 |
| BTC Daily | lstm | LSTM | 724 | 312 | 412 | 43.09% | 37.08% | 41.67% | 6.91 pp | -100 | 43 | -2.33 |
| BTC Daily | rf | RandomForest | 724 | 309 | 415 | 42.68% | 40.83% | 43.54% | 7.32 pp | -106 | 43 | -2.47 |
| BTC Daily | xgb | XGBoost | 734 | 291 | 443 | 39.65% | 36.67% | 38.75% | 10.35 pp | -152 | 43 | -3.53 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 496 | 240 | 256 | 48.39% | 44.58% | 48.33% | 1.61 pp | -16 | 48 | -0.33 |
| BTC Market Hours | nn | NN | 496 | 234 | 262 | 47.18% | 50.00% | 47.71% | 2.82 pp | -28 | 48 | -0.58 |
| BTC Market Hours | transformer | Transformer | 496 | 233 | 263 | 46.98% | 44.17% | 47.71% | 3.02 pp | -30 | 48 | -0.62 |
| BTC Market Hours | lstm | LSTM | 496 | 214 | 282 | 43.15% | 40.83% | 43.12% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | rf | RandomForest | 496 | 214 | 282 | 43.15% | 43.75% | 43.54% | 6.85 pp | -68 | 48 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 496 | 204 | 292 | 41.13% | 41.25% | 41.25% | 8.87 pp | -88 | 48 | -1.83 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 550 | 256 | 294 | 46.55% | 49.17% | 47.50% | 3.45 pp | -38 | 47 | -0.81 |
| BTC Market Hours Daily | nn | NN | 550 | 254 | 296 | 46.18% | 45.00% | 47.50% | 3.82 pp | -42 | 47 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 550 | 253 | 297 | 46.00% | 48.75% | 46.88% | 4.00 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 550 | 229 | 321 | 41.64% | 42.50% | 41.04% | 8.36 pp | -92 | 47 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 550 | 222 | 328 | 40.36% | 38.75% | 40.62% | 9.64 pp | -106 | 47 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 550 | 220 | 330 | 40.00% | 41.25% | 39.38% | 10.00 pp | -110 | 47 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
