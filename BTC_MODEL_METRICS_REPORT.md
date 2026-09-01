# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T03:52:26.307261+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1173 | 885 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1049 | 684 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 678 | 446 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 680 | 500 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 97 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 18:00:00+00:00 | 97 | 8 | 89 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 446 | 218 | 228 | 48.88% | 45.42% | 48.88% | 1.12 pp | -10 | 44 | -0.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 674 | 331 | 343 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| BTC Daily | transformer | Transformer | 674 | 327 | 347 | 48.52% | 46.25% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 446 | 210 | 236 | 47.09% | 48.75% | 47.09% | 2.91 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 446 | 205 | 241 | 45.96% | 40.83% | 45.96% | 4.04 pp | -36 | 44 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 500 | 231 | 269 | 46.20% | 47.08% | 46.67% | 3.80 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | nn | NN | 500 | 229 | 271 | 45.80% | 43.75% | 46.67% | 4.20 pp | -42 | 44 | -0.95 |
| BTC Hourly | transformer | Transformer | 851 | 402 | 449 | 47.24% | 47.08% | 46.88% | 2.76 pp | -47 | 45 | -1.04 |
| BTC Daily | nn | NN | 674 | 315 | 359 | 46.74% | 42.92% | 48.96% | 3.26 pp | -44 | 41 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 500 | 226 | 274 | 45.20% | 45.00% | 45.21% | 4.80 pp | -48 | 44 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 851 | 400 | 451 | 47.00% | 44.58% | 46.67% | 3.00 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 446 | 194 | 252 | 43.50% | 43.75% | 43.50% | 6.50 pp | -58 | 44 | -1.32 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 446 | 191 | 255 | 42.83% | 40.42% | 42.83% | 7.17 pp | -64 | 44 | -1.45 |
| BTC Hourly | nn | NN | 851 | 385 | 466 | 45.24% | 45.00% | 44.79% | 4.76 pp | -81 | 45 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 500 | 208 | 292 | 41.60% | 42.08% | 41.88% | 8.40 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 446 | 180 | 266 | 40.36% | 38.75% | 40.36% | 9.64 pp | -86 | 44 | -1.95 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 851 | 379 | 472 | 44.54% | 43.33% | 43.96% | 5.46 pp | -93 | 45 | -2.07 |
| BTC Daily | lstm | LSTM | 674 | 294 | 380 | 43.62% | 38.75% | 42.92% | 6.38 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 500 | 200 | 300 | 40.00% | 37.50% | 40.62% | 10.00 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 674 | 290 | 384 | 43.03% | 41.25% | 43.75% | 6.97 pp | -94 | 41 | -2.29 |
| BTC Market Hours Daily | xgb | XGBoost | 500 | 196 | 304 | 39.20% | 36.25% | 38.96% | 10.80 pp | -108 | 44 | -2.45 |
| BTC Hourly | lstm | LSTM | 851 | 363 | 488 | 42.66% | 38.33% | 42.08% | 7.34 pp | -125 | 45 | -2.78 |
| BTC Hourly | xgb | XGBoost | 851 | 358 | 493 | 42.07% | 40.00% | 42.50% | 7.93 pp | -135 | 45 | -3.00 |
| BTC Daily | xgb | XGBoost | 684 | 272 | 412 | 39.77% | 35.00% | 39.58% | 10.23 pp | -140 | 41 | -3.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 851 | 402 | 449 | 47.24% | 47.08% | 46.88% | 2.76 pp | -47 | 45 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 851 | 400 | 451 | 47.00% | 44.58% | 46.67% | 3.00 pp | -51 | 45 | -1.13 |
| BTC Hourly | nn | NN | 851 | 385 | 466 | 45.24% | 45.00% | 44.79% | 4.76 pp | -81 | 45 | -1.80 |
| BTC Hourly | rf | RandomForest | 851 | 379 | 472 | 44.54% | 43.33% | 43.96% | 5.46 pp | -93 | 45 | -2.07 |
| BTC Hourly | lstm | LSTM | 851 | 363 | 488 | 42.66% | 38.33% | 42.08% | 7.34 pp | -125 | 45 | -2.78 |
| BTC Hourly | xgb | XGBoost | 851 | 358 | 493 | 42.07% | 40.00% | 42.50% | 7.93 pp | -135 | 45 | -3.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 674 | 331 | 343 | 49.11% | 48.33% | 50.00% | 0.89 pp | -12 | 41 | -0.29 |
| BTC Daily | transformer | Transformer | 674 | 327 | 347 | 48.52% | 46.25% | 49.58% | 1.48 pp | -20 | 41 | -0.49 |
| BTC Daily | nn | NN | 674 | 315 | 359 | 46.74% | 42.92% | 48.96% | 3.26 pp | -44 | 41 | -1.07 |
| BTC Daily | lstm | LSTM | 674 | 294 | 380 | 43.62% | 38.75% | 42.92% | 6.38 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 674 | 290 | 384 | 43.03% | 41.25% | 43.75% | 6.97 pp | -94 | 41 | -2.29 |
| BTC Daily | xgb | XGBoost | 684 | 272 | 412 | 39.77% | 35.00% | 39.58% | 10.23 pp | -140 | 41 | -3.41 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 446 | 218 | 228 | 48.88% | 45.42% | 48.88% | 1.12 pp | -10 | 44 | -0.23 |
| BTC Market Hours | nn | NN | 446 | 210 | 236 | 47.09% | 48.75% | 47.09% | 2.91 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 446 | 205 | 241 | 45.96% | 40.83% | 45.96% | 4.04 pp | -36 | 44 | -0.82 |
| BTC Market Hours | rf | RandomForest | 446 | 194 | 252 | 43.50% | 43.75% | 43.50% | 6.50 pp | -58 | 44 | -1.32 |
| BTC Market Hours | lstm | LSTM | 446 | 191 | 255 | 42.83% | 40.42% | 42.83% | 7.17 pp | -64 | 44 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 446 | 180 | 266 | 40.36% | 38.75% | 40.36% | 9.64 pp | -86 | 44 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 500 | 231 | 269 | 46.20% | 47.08% | 46.67% | 3.80 pp | -38 | 44 | -0.86 |
| BTC Market Hours Daily | nn | NN | 500 | 229 | 271 | 45.80% | 43.75% | 46.67% | 4.20 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 500 | 226 | 274 | 45.20% | 45.00% | 45.21% | 4.80 pp | -48 | 44 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 500 | 208 | 292 | 41.60% | 42.08% | 41.88% | 8.40 pp | -84 | 44 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 500 | 200 | 300 | 40.00% | 37.50% | 40.62% | 10.00 pp | -100 | 44 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 500 | 196 | 304 | 39.20% | 36.25% | 38.96% | 10.80 pp | -108 | 44 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 52 | 45 | 53.61% | 53.61% | 53.61% | 3.61 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 48 | 49 | 49.48% | 49.48% | 49.48% | 0.52 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 42 | 55 | 43.30% | 43.30% | 43.30% | 6.70 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
