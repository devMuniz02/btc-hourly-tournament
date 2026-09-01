# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T03:17:55.682411+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1172 | 884 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1048 | 683 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 677 | 445 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 679 | 499 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 97 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T18:00:00+00:00 | 98 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 445 | 218 | 227 | 48.99% | 45.83% | 48.99% | 1.01 pp | -9 | 44 | -0.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 673 | 330 | 343 | 49.03% | 47.92% | 49.79% | 0.97 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 673 | 326 | 347 | 48.44% | 45.83% | 49.58% | 1.56 pp | -21 | 41 | -0.51 |
| Consolidated Hourly | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 445 | 210 | 235 | 47.19% | 48.75% | 47.19% | 2.81 pp | -25 | 44 | -0.57 |
| BTC Market Hours | transformer | Transformer | 445 | 204 | 241 | 45.84% | 40.42% | 45.84% | 4.16 pp | -37 | 44 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 499 | 230 | 269 | 46.09% | 46.67% | 46.67% | 3.91 pp | -39 | 44 | -0.89 |
| BTC Market Hours Daily | nn | NN | 499 | 229 | 270 | 45.89% | 43.75% | 46.88% | 4.11 pp | -41 | 44 | -0.93 |
| Consolidated Hourly | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Daily | nn | NN | 673 | 315 | 358 | 46.81% | 42.92% | 49.17% | 3.19 pp | -43 | 41 | -1.05 |
| BTC Hourly | transformer | Transformer | 850 | 401 | 449 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 499 | 225 | 274 | 45.09% | 44.58% | 45.21% | 4.91 pp | -49 | 44 | -1.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 850 | 399 | 451 | 46.94% | 44.58% | 46.46% | 3.06 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 445 | 193 | 252 | 43.37% | 43.33% | 43.37% | 6.63 pp | -59 | 44 | -1.34 |
| BTC Market Hours | lstm | LSTM | 445 | 191 | 254 | 42.92% | 40.83% | 42.92% | 7.08 pp | -63 | 44 | -1.43 |
| BTC Hourly | nn | NN | 850 | 384 | 466 | 45.18% | 44.58% | 44.58% | 4.82 pp | -82 | 45 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 499 | 207 | 292 | 41.48% | 41.67% | 41.88% | 8.52 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 445 | 179 | 266 | 40.22% | 38.33% | 40.22% | 9.78 pp | -87 | 44 | -1.98 |
| Consolidated Market Hours | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 850 | 378 | 472 | 44.47% | 43.33% | 43.75% | 5.53 pp | -94 | 45 | -2.09 |
| BTC Daily | lstm | LSTM | 673 | 293 | 380 | 43.54% | 38.33% | 42.92% | 6.46 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 499 | 199 | 300 | 39.88% | 37.50% | 40.62% | 10.12 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 673 | 289 | 384 | 42.94% | 41.25% | 43.75% | 7.06 pp | -95 | 41 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 499 | 195 | 304 | 39.08% | 36.25% | 38.96% | 10.92 pp | -109 | 44 | -2.48 |
| BTC Hourly | lstm | LSTM | 850 | 363 | 487 | 42.71% | 38.75% | 42.08% | 7.29 pp | -124 | 45 | -2.76 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Hourly | xgb | XGBoost | 850 | 357 | 493 | 42.00% | 40.00% | 42.29% | 8.00 pp | -136 | 45 | -3.02 |
| BTC Daily | xgb | XGBoost | 683 | 272 | 411 | 39.82% | 35.00% | 39.58% | 10.18 pp | -139 | 41 | -3.39 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 850 | 401 | 449 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 850 | 399 | 451 | 46.94% | 44.58% | 46.46% | 3.06 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 850 | 384 | 466 | 45.18% | 44.58% | 44.58% | 4.82 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 850 | 378 | 472 | 44.47% | 43.33% | 43.75% | 5.53 pp | -94 | 45 | -2.09 |
| BTC Hourly | lstm | LSTM | 850 | 363 | 487 | 42.71% | 38.75% | 42.08% | 7.29 pp | -124 | 45 | -2.76 |
| BTC Hourly | xgb | XGBoost | 850 | 357 | 493 | 42.00% | 40.00% | 42.29% | 8.00 pp | -136 | 45 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 673 | 330 | 343 | 49.03% | 47.92% | 49.79% | 0.97 pp | -13 | 41 | -0.32 |
| BTC Daily | transformer | Transformer | 673 | 326 | 347 | 48.44% | 45.83% | 49.58% | 1.56 pp | -21 | 41 | -0.51 |
| BTC Daily | nn | NN | 673 | 315 | 358 | 46.81% | 42.92% | 49.17% | 3.19 pp | -43 | 41 | -1.05 |
| BTC Daily | lstm | LSTM | 673 | 293 | 380 | 43.54% | 38.33% | 42.92% | 6.46 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 673 | 289 | 384 | 42.94% | 41.25% | 43.75% | 7.06 pp | -95 | 41 | -2.32 |
| BTC Daily | xgb | XGBoost | 683 | 272 | 411 | 39.82% | 35.00% | 39.58% | 10.18 pp | -139 | 41 | -3.39 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 445 | 218 | 227 | 48.99% | 45.83% | 48.99% | 1.01 pp | -9 | 44 | -0.20 |
| BTC Market Hours | nn | NN | 445 | 210 | 235 | 47.19% | 48.75% | 47.19% | 2.81 pp | -25 | 44 | -0.57 |
| BTC Market Hours | transformer | Transformer | 445 | 204 | 241 | 45.84% | 40.42% | 45.84% | 4.16 pp | -37 | 44 | -0.84 |
| BTC Market Hours | rf | RandomForest | 445 | 193 | 252 | 43.37% | 43.33% | 43.37% | 6.63 pp | -59 | 44 | -1.34 |
| BTC Market Hours | lstm | LSTM | 445 | 191 | 254 | 42.92% | 40.83% | 42.92% | 7.08 pp | -63 | 44 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 445 | 179 | 266 | 40.22% | 38.33% | 40.22% | 9.78 pp | -87 | 44 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 499 | 230 | 269 | 46.09% | 46.67% | 46.67% | 3.91 pp | -39 | 44 | -0.89 |
| BTC Market Hours Daily | nn | NN | 499 | 229 | 270 | 45.89% | 43.75% | 46.88% | 4.11 pp | -41 | 44 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 499 | 225 | 274 | 45.09% | 44.58% | 45.21% | 4.91 pp | -49 | 44 | -1.11 |
| BTC Market Hours Daily | rf | RandomForest | 499 | 207 | 292 | 41.48% | 41.67% | 41.88% | 8.52 pp | -85 | 44 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 499 | 199 | 300 | 39.88% | 37.50% | 40.62% | 10.12 pp | -101 | 44 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 499 | 195 | 304 | 39.08% | 36.25% | 38.96% | 10.92 pp | -109 | 44 | -2.48 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 97 | 50 | 47 | 51.55% | 51.55% | 51.55% | 1.55 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
