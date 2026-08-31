# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T18:14:23.236149+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1166 | 878 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1042 | 677 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 17:00:00+00:00 | 664 | 439 | 224 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 17:00:00+00:00 | 666 | 493 | 171 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 91 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 91 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 91 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T15:00:00+00:00 | 92 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 91 | 48 | 43 | 52.75% | 52.75% | 52.75% | 2.75 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 48 | 43 | 52.75% | 52.75% | 52.75% | 2.75 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | xgb | XGBoost | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 439 | 215 | 224 | 48.97% | 44.58% | 48.97% | 1.03 pp | -9 | 43 | -0.21 |
| Consolidated Hourly | nn | NN | 91 | 44 | 47 | 48.35% | 48.35% | 48.35% | 1.65 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 44 | 47 | 48.35% | 48.35% | 48.35% | 1.65 pp | -3 | 9 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 667 | 326 | 341 | 48.88% | 47.08% | 49.79% | 1.12 pp | -15 | 41 | -0.37 |
| Consolidated Hourly | lstm | LSTM | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| BTC Daily | transformer | Transformer | 667 | 322 | 345 | 48.28% | 45.83% | 49.58% | 1.72 pp | -23 | 41 | -0.56 |
| BTC Market Hours | nn | NN | 439 | 207 | 232 | 47.15% | 48.75% | 47.15% | 2.85 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 439 | 202 | 237 | 46.01% | 41.25% | 46.01% | 3.99 pp | -35 | 43 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 493 | 228 | 265 | 46.25% | 47.08% | 46.46% | 3.75 pp | -37 | 43 | -0.86 |
| BTC Daily | nn | NN | 667 | 313 | 354 | 46.93% | 43.33% | 49.38% | 3.07 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | nn | NN | 493 | 225 | 268 | 45.64% | 43.75% | 46.25% | 4.36 pp | -43 | 43 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 844 | 399 | 445 | 47.27% | 47.92% | 47.08% | 2.73 pp | -46 | 45 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 493 | 224 | 269 | 45.44% | 45.42% | 45.21% | 4.56 pp | -45 | 43 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 844 | 396 | 448 | 46.92% | 44.17% | 46.67% | 3.08 pp | -52 | 45 | -1.16 |
| Consolidated Hourly | transformer | Transformer | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| BTC Market Hours | lstm | LSTM | 439 | 188 | 251 | 42.82% | 41.67% | 42.82% | 7.18 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 439 | 188 | 251 | 42.82% | 42.92% | 42.82% | 7.18 pp | -63 | 43 | -1.47 |
| BTC Hourly | nn | NN | 844 | 381 | 463 | 45.14% | 44.17% | 44.58% | 4.86 pp | -82 | 45 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 493 | 203 | 290 | 41.18% | 41.25% | 41.46% | 8.82 pp | -87 | 43 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 439 | 176 | 263 | 40.09% | 38.33% | 40.09% | 9.91 pp | -87 | 43 | -2.02 |
| BTC Daily | lstm | LSTM | 667 | 292 | 375 | 43.78% | 39.17% | 43.12% | 6.22 pp | -83 | 41 | -2.02 |
| BTC Hourly | rf | RandomForest | 844 | 376 | 468 | 44.55% | 43.75% | 43.96% | 5.45 pp | -92 | 45 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 493 | 197 | 296 | 39.96% | 37.92% | 40.62% | 10.04 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 667 | 285 | 382 | 42.73% | 40.83% | 43.75% | 7.27 pp | -97 | 41 | -2.37 |
| BTC Market Hours Daily | xgb | XGBoost | 493 | 193 | 300 | 39.15% | 36.25% | 39.17% | 10.85 pp | -107 | 43 | -2.49 |
| BTC Hourly | lstm | LSTM | 844 | 362 | 482 | 42.89% | 40.00% | 42.29% | 7.11 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 844 | 356 | 488 | 42.18% | 40.00% | 42.50% | 7.82 pp | -132 | 45 | -2.93 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 677 | 269 | 408 | 39.73% | 34.17% | 39.79% | 10.27 pp | -139 | 41 | -3.39 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 844 | 399 | 445 | 47.27% | 47.92% | 47.08% | 2.73 pp | -46 | 45 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 844 | 396 | 448 | 46.92% | 44.17% | 46.67% | 3.08 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 844 | 381 | 463 | 45.14% | 44.17% | 44.58% | 4.86 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 844 | 376 | 468 | 44.55% | 43.75% | 43.96% | 5.45 pp | -92 | 45 | -2.04 |
| BTC Hourly | lstm | LSTM | 844 | 362 | 482 | 42.89% | 40.00% | 42.29% | 7.11 pp | -120 | 45 | -2.67 |
| BTC Hourly | xgb | XGBoost | 844 | 356 | 488 | 42.18% | 40.00% | 42.50% | 7.82 pp | -132 | 45 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 667 | 326 | 341 | 48.88% | 47.08% | 49.79% | 1.12 pp | -15 | 41 | -0.37 |
| BTC Daily | transformer | Transformer | 667 | 322 | 345 | 48.28% | 45.83% | 49.58% | 1.72 pp | -23 | 41 | -0.56 |
| BTC Daily | nn | NN | 667 | 313 | 354 | 46.93% | 43.33% | 49.38% | 3.07 pp | -41 | 41 | -1.00 |
| BTC Daily | lstm | LSTM | 667 | 292 | 375 | 43.78% | 39.17% | 43.12% | 6.22 pp | -83 | 41 | -2.02 |
| BTC Daily | rf | RandomForest | 667 | 285 | 382 | 42.73% | 40.83% | 43.75% | 7.27 pp | -97 | 41 | -2.37 |
| BTC Daily | xgb | XGBoost | 677 | 269 | 408 | 39.73% | 34.17% | 39.79% | 10.27 pp | -139 | 41 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 439 | 215 | 224 | 48.97% | 44.58% | 48.97% | 1.03 pp | -9 | 43 | -0.21 |
| BTC Market Hours | nn | NN | 439 | 207 | 232 | 47.15% | 48.75% | 47.15% | 2.85 pp | -25 | 43 | -0.58 |
| BTC Market Hours | transformer | Transformer | 439 | 202 | 237 | 46.01% | 41.25% | 46.01% | 3.99 pp | -35 | 43 | -0.81 |
| BTC Market Hours | lstm | LSTM | 439 | 188 | 251 | 42.82% | 41.67% | 42.82% | 7.18 pp | -63 | 43 | -1.47 |
| BTC Market Hours | rf | RandomForest | 439 | 188 | 251 | 42.82% | 42.92% | 42.82% | 7.18 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 439 | 176 | 263 | 40.09% | 38.33% | 40.09% | 9.91 pp | -87 | 43 | -2.02 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 493 | 228 | 265 | 46.25% | 47.08% | 46.46% | 3.75 pp | -37 | 43 | -0.86 |
| BTC Market Hours Daily | nn | NN | 493 | 225 | 268 | 45.64% | 43.75% | 46.25% | 4.36 pp | -43 | 43 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 493 | 224 | 269 | 45.44% | 45.42% | 45.21% | 4.56 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 493 | 203 | 290 | 41.18% | 41.25% | 41.46% | 8.82 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 493 | 197 | 296 | 39.96% | 37.92% | 40.62% | 10.04 pp | -99 | 43 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 493 | 193 | 300 | 39.15% | 36.25% | 39.17% | 10.85 pp | -107 | 43 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 91 | 48 | 43 | 52.75% | 52.75% | 52.75% | 2.75 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | xgb | XGBoost | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 91 | 44 | 47 | 48.35% | 48.35% | 48.35% | 1.65 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 48 | 43 | 52.75% | 52.75% | 52.75% | 2.75 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 44 | 47 | 48.35% | 48.35% | 48.35% | 1.65 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 6 | 3 | 3 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
