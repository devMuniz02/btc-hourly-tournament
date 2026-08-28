# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T17:08:02.538796+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 823 | 296 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 987 | 622 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 569 | 384 | 184 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 571 | 438 | 131 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 17:00:00+00:00 | 45 | 45 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 17:00:00+00:00 | 45 | 45 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 17:00:00+00:00 | 45 | 1 | 44 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 17:00:00+00:00 | 45 | 1 | 44 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 45 | 26 | 19 | 57.78% | 57.78% | 57.78% | 7.78 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 45 | 26 | 19 | 57.78% | 57.78% | 57.78% | 7.78 pp | 7 | 5 | 1.40 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 384 | 189 | 195 | 49.22% | 47.50% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| BTC Daily | transformer | Transformer | 612 | 301 | 311 | 49.18% | 49.58% | 50.21% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 612 | 300 | 312 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| Consolidated Hourly | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| BTC Market Hours | nn | NN | 384 | 178 | 206 | 46.35% | 48.75% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 384 | 178 | 206 | 46.35% | 43.33% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 438 | 201 | 237 | 45.89% | 47.50% | 45.89% | 4.11 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 438 | 200 | 238 | 45.66% | 45.83% | 45.66% | 4.34 pp | -38 | 39 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 789 | 373 | 416 | 47.28% | 45.00% | 47.29% | 2.72 pp | -43 | 43 | -1.00 |
| BTC Daily | nn | NN | 612 | 287 | 325 | 46.90% | 43.75% | 48.54% | 3.10 pp | -38 | 38 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 438 | 198 | 240 | 45.21% | 45.83% | 45.21% | 4.79 pp | -42 | 39 | -1.08 |
| BTC Hourly | transformer | Transformer | 789 | 371 | 418 | 47.02% | 44.17% | 46.46% | 2.98 pp | -47 | 43 | -1.09 |
| BTC Market Hours | lstm | LSTM | 384 | 165 | 219 | 42.97% | 43.33% | 42.97% | 7.03 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 384 | 164 | 220 | 42.71% | 40.83% | 42.71% | 7.29 pp | -56 | 39 | -1.44 |
| BTC Daily | lstm | LSTM | 612 | 274 | 338 | 44.77% | 44.17% | 44.79% | 5.23 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 789 | 356 | 433 | 45.12% | 40.83% | 45.83% | 4.88 pp | -77 | 43 | -1.79 |
| BTC Market Hours | xgb | XGBoost | 384 | 155 | 229 | 40.36% | 38.33% | 40.36% | 9.64 pp | -74 | 39 | -1.90 |
| BTC Hourly | rf | RandomForest | 789 | 350 | 439 | 44.36% | 42.50% | 43.75% | 5.64 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 438 | 178 | 260 | 40.64% | 39.17% | 40.64% | 9.36 pp | -82 | 39 | -2.10 |
| BTC Hourly | lstm | LSTM | 789 | 348 | 441 | 44.11% | 44.17% | 45.42% | 5.89 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 612 | 263 | 349 | 42.97% | 42.92% | 43.75% | 7.03 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 438 | 173 | 265 | 39.50% | 37.50% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 438 | 173 | 265 | 39.50% | 38.33% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| Consolidated Hourly | nn | NN | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 5 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 5 | -2.60 |
| BTC Hourly | xgb | XGBoost | 789 | 336 | 453 | 42.59% | 39.17% | 43.96% | 7.41 pp | -117 | 43 | -2.72 |
| BTC Daily | xgb | XGBoost | 622 | 247 | 375 | 39.71% | 33.33% | 39.79% | 10.29 pp | -128 | 38 | -3.37 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 789 | 373 | 416 | 47.28% | 45.00% | 47.29% | 2.72 pp | -43 | 43 | -1.00 |
| BTC Hourly | transformer | Transformer | 789 | 371 | 418 | 47.02% | 44.17% | 46.46% | 2.98 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 789 | 356 | 433 | 45.12% | 40.83% | 45.83% | 4.88 pp | -77 | 43 | -1.79 |
| BTC Hourly | rf | RandomForest | 789 | 350 | 439 | 44.36% | 42.50% | 43.75% | 5.64 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 789 | 348 | 441 | 44.11% | 44.17% | 45.42% | 5.89 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 789 | 336 | 453 | 42.59% | 39.17% | 43.96% | 7.41 pp | -117 | 43 | -2.72 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 612 | 301 | 311 | 49.18% | 49.58% | 50.21% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 612 | 300 | 312 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 612 | 287 | 325 | 46.90% | 43.75% | 48.54% | 3.10 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 612 | 274 | 338 | 44.77% | 44.17% | 44.79% | 5.23 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 612 | 263 | 349 | 42.97% | 42.92% | 43.75% | 7.03 pp | -86 | 38 | -2.26 |
| BTC Daily | xgb | XGBoost | 622 | 247 | 375 | 39.71% | 33.33% | 39.79% | 10.29 pp | -128 | 38 | -3.37 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 384 | 189 | 195 | 49.22% | 47.50% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| BTC Market Hours | nn | NN | 384 | 178 | 206 | 46.35% | 48.75% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 384 | 178 | 206 | 46.35% | 43.33% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 384 | 165 | 219 | 42.97% | 43.33% | 42.97% | 7.03 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 384 | 164 | 220 | 42.71% | 40.83% | 42.71% | 7.29 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 384 | 155 | 229 | 40.36% | 38.33% | 40.36% | 9.64 pp | -74 | 39 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 438 | 201 | 237 | 45.89% | 47.50% | 45.89% | 4.11 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 438 | 200 | 238 | 45.66% | 45.83% | 45.66% | 4.34 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 438 | 198 | 240 | 45.21% | 45.83% | 45.21% | 4.79 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 438 | 178 | 260 | 40.64% | 39.17% | 40.64% | 9.36 pp | -82 | 39 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 438 | 173 | 265 | 39.50% | 37.50% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 438 | 173 | 265 | 39.50% | 38.33% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 45 | 26 | 19 | 57.78% | 57.78% | 57.78% | 7.78 pp | 7 | 5 | 1.40 |
| Consolidated Hourly | lstm | LSTM | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 5 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 45 | 26 | 19 | 57.78% | 57.78% | 57.78% | 7.78 pp | 7 | 5 | 1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 45 | 24 | 21 | 53.33% | 53.33% | 53.33% | 3.33 pp | 3 | 5 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
