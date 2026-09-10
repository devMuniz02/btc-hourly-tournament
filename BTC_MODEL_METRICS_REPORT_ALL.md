# BTC Model Metrics Report - All Rows

Generated at: 2026-09-10T21:23:10.069308+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1330 | 1042 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1206 | 841 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 961 | 603 | 357 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-10 20:00:00+00:00 | 962 | 656 | 304 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 243 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 19:00:00+00:00 | 243 | 87 | 156 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 603 | 290 | 313 | 48.09% | 46.67% | 47.29% | 1.91 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 603 | 287 | 316 | 47.60% | 50.83% | 48.96% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 603 | 282 | 321 | 46.77% | 45.83% | 46.04% | 3.23 pp | -39 | 56 | -0.70 |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | nn | NN | 656 | 307 | 349 | 46.80% | 48.33% | 47.50% | 3.20 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 656 | 306 | 350 | 46.65% | 48.33% | 46.88% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 656 | 306 | 350 | 46.65% | 47.92% | 47.92% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Daily | mlp_sklearn | MLPClassifier | 831 | 396 | 435 | 47.65% | 44.58% | 45.83% | 2.35 pp | -39 | 47 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1008 | 478 | 530 | 47.42% | 48.33% | 45.83% | 2.58 pp | -52 | 52 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| BTC Daily | nn | NN | 831 | 387 | 444 | 46.57% | 45.00% | 45.21% | 3.43 pp | -57 | 47 | -1.21 |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| BTC Hourly | transformer | Transformer | 1008 | 470 | 538 | 46.63% | 46.25% | 44.79% | 3.37 pp | -68 | 52 | -1.31 |
| BTC Daily | transformer | Transformer | 831 | 384 | 447 | 46.21% | 37.92% | 45.00% | 3.79 pp | -63 | 47 | -1.34 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| BTC Market Hours | lstm | LSTM | 603 | 258 | 345 | 42.79% | 42.50% | 42.71% | 7.21 pp | -87 | 56 | -1.55 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours | rf | RandomForest | 603 | 257 | 346 | 42.62% | 42.92% | 42.08% | 7.38 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 603 | 256 | 347 | 42.45% | 45.00% | 43.12% | 7.55 pp | -91 | 56 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| BTC Market Hours Daily | rf | RandomForest | 656 | 271 | 385 | 41.31% | 42.08% | 41.25% | 8.69 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 656 | 268 | 388 | 40.85% | 42.50% | 40.42% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 656 | 268 | 388 | 40.85% | 42.50% | 40.62% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Hourly | nn | NN | 1008 | 444 | 564 | 44.05% | 42.08% | 41.25% | 5.95 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1008 | 444 | 564 | 44.05% | 41.67% | 43.12% | 5.95 pp | -120 | 52 | -2.31 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| BTC Daily | lstm | LSTM | 831 | 350 | 481 | 42.12% | 35.42% | 40.00% | 7.88 pp | -131 | 47 | -2.79 |
| BTC Hourly | lstm | LSTM | 1008 | 426 | 582 | 42.26% | 36.25% | 40.00% | 7.74 pp | -156 | 52 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| BTC Daily | rf | RandomForest | 831 | 344 | 487 | 41.40% | 36.67% | 40.62% | 8.60 pp | -143 | 47 | -3.04 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| BTC Hourly | xgb | XGBoost | 1008 | 414 | 594 | 41.07% | 35.42% | 38.33% | 8.93 pp | -180 | 52 | -3.46 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 841 | 332 | 509 | 39.48% | 37.50% | 37.08% | 10.52 pp | -177 | 47 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1008 | 478 | 530 | 47.42% | 48.33% | 45.83% | 2.58 pp | -52 | 52 | -1.00 |
| BTC Hourly | transformer | Transformer | 1008 | 470 | 538 | 46.63% | 46.25% | 44.79% | 3.37 pp | -68 | 52 | -1.31 |
| BTC Hourly | nn | NN | 1008 | 444 | 564 | 44.05% | 42.08% | 41.25% | 5.95 pp | -120 | 52 | -2.31 |
| BTC Hourly | rf | RandomForest | 1008 | 444 | 564 | 44.05% | 41.67% | 43.12% | 5.95 pp | -120 | 52 | -2.31 |
| BTC Hourly | lstm | LSTM | 1008 | 426 | 582 | 42.26% | 36.25% | 40.00% | 7.74 pp | -156 | 52 | -3.00 |
| BTC Hourly | xgb | XGBoost | 1008 | 414 | 594 | 41.07% | 35.42% | 38.33% | 8.93 pp | -180 | 52 | -3.46 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 831 | 396 | 435 | 47.65% | 44.58% | 45.83% | 2.35 pp | -39 | 47 | -0.83 |
| BTC Daily | nn | NN | 831 | 387 | 444 | 46.57% | 45.00% | 45.21% | 3.43 pp | -57 | 47 | -1.21 |
| BTC Daily | transformer | Transformer | 831 | 384 | 447 | 46.21% | 37.92% | 45.00% | 3.79 pp | -63 | 47 | -1.34 |
| BTC Daily | lstm | LSTM | 831 | 350 | 481 | 42.12% | 35.42% | 40.00% | 7.88 pp | -131 | 47 | -2.79 |
| BTC Daily | rf | RandomForest | 831 | 344 | 487 | 41.40% | 36.67% | 40.62% | 8.60 pp | -143 | 47 | -3.04 |
| BTC Daily | xgb | XGBoost | 841 | 332 | 509 | 39.48% | 37.50% | 37.08% | 10.52 pp | -177 | 47 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 603 | 290 | 313 | 48.09% | 46.67% | 47.29% | 1.91 pp | -23 | 56 | -0.41 |
| BTC Market Hours | nn | NN | 603 | 287 | 316 | 47.60% | 50.83% | 48.96% | 2.40 pp | -29 | 56 | -0.52 |
| BTC Market Hours | transformer | Transformer | 603 | 282 | 321 | 46.77% | 45.83% | 46.04% | 3.23 pp | -39 | 56 | -0.70 |
| BTC Market Hours | lstm | LSTM | 603 | 258 | 345 | 42.79% | 42.50% | 42.71% | 7.21 pp | -87 | 56 | -1.55 |
| BTC Market Hours | rf | RandomForest | 603 | 257 | 346 | 42.62% | 42.92% | 42.08% | 7.38 pp | -89 | 56 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 603 | 256 | 347 | 42.45% | 45.00% | 43.12% | 7.55 pp | -91 | 56 | -1.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 656 | 307 | 349 | 46.80% | 48.33% | 47.50% | 3.20 pp | -42 | 56 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 656 | 306 | 350 | 46.65% | 48.33% | 46.88% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | transformer | Transformer | 656 | 306 | 350 | 46.65% | 47.92% | 47.92% | 3.35 pp | -44 | 56 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 656 | 271 | 385 | 41.31% | 42.08% | 41.25% | 8.69 pp | -114 | 56 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 656 | 268 | 388 | 40.85% | 42.50% | 40.42% | 9.15 pp | -120 | 56 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 656 | 268 | 388 | 40.85% | 42.50% | 40.62% | 9.15 pp | -120 | 56 | -2.14 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 243 | 116 | 127 | 47.74% | 47.92% | 47.74% | 2.26 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 243 | 113 | 130 | 46.50% | 46.25% | 46.50% | 3.50 pp | -17 | 15 | -1.13 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 243 | 111 | 132 | 45.68% | 45.83% | 45.68% | 4.32 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 243 | 109 | 134 | 44.86% | 44.58% | 44.86% | 5.14 pp | -25 | 15 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 243 | 101 | 142 | 41.56% | 41.25% | 41.56% | 8.44 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 243 | 97 | 146 | 39.92% | 40.00% | 39.92% | 10.08 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 7 | -1.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 35 | 52 | 40.23% | 40.23% | 40.23% | 9.77 pp | -17 | 7 | -2.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours Daily | nn | NN | 87 | 31 | 56 | 35.63% | 35.63% | 35.63% | 14.37 pp | -25 | 7 | -3.57 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
