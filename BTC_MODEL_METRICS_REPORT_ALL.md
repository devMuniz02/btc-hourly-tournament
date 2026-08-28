# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T03:23:36.673049+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 812 | 307 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 976 | 611 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 553 | 373 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 555 | 427 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 19:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 19:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 19:00:00+00:00 | 36 | 1 | 35 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 19:00:00+00:00 | 36 | 1 | 35 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 36 | 21 | 15 | 58.33% | 58.33% | 58.33% | 8.33 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 21 | 15 | 58.33% | 58.33% | 58.33% | 8.33 pp | 6 | 4 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| BTC Daily | transformer | Transformer | 601 | 297 | 304 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 601 | 283 | 318 | 47.09% | 45.00% | 48.33% | 2.91 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 427 | 195 | 232 | 45.67% | 45.83% | 45.67% | 4.33 pp | -37 | 38 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 778 | 368 | 410 | 47.30% | 44.58% | 47.50% | 2.70 pp | -42 | 42 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 427 | 194 | 233 | 45.43% | 46.67% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 427 | 194 | 233 | 45.43% | 47.92% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 778 | 365 | 413 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 601 | 268 | 333 | 44.59% | 42.92% | 44.79% | 5.41 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 778 | 349 | 429 | 44.86% | 40.42% | 45.62% | 5.14 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 778 | 348 | 430 | 44.73% | 43.33% | 44.38% | 5.27 pp | -82 | 42 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |
| Consolidated Hourly | xgb | XGBoost | 36 | 14 | 22 | 38.89% | 38.89% | 38.89% | 11.11 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 14 | 22 | 38.89% | 38.89% | 38.89% | 11.11 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 601 | 261 | 340 | 43.43% | 45.00% | 44.17% | 6.57 pp | -79 | 38 | -2.08 |
| BTC Market Hours Daily | rf | RandomForest | 427 | 173 | 254 | 40.52% | 40.42% | 40.52% | 9.48 pp | -81 | 38 | -2.13 |
| BTC Hourly | lstm | LSTM | 778 | 344 | 434 | 44.22% | 43.33% | 45.83% | 5.78 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 778 | 333 | 445 | 42.80% | 40.00% | 43.96% | 7.20 pp | -112 | 42 | -2.67 |
| BTC Daily | xgb | XGBoost | 611 | 246 | 365 | 40.26% | 35.83% | 40.42% | 9.74 pp | -119 | 38 | -3.13 |
| Consolidated Hourly | nn | NN | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 778 | 368 | 410 | 47.30% | 44.58% | 47.50% | 2.70 pp | -42 | 42 | -1.00 |
| BTC Hourly | transformer | Transformer | 778 | 365 | 413 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 42 | -1.14 |
| BTC Hourly | nn | NN | 778 | 349 | 429 | 44.86% | 40.42% | 45.62% | 5.14 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 778 | 348 | 430 | 44.73% | 43.33% | 44.38% | 5.27 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 778 | 344 | 434 | 44.22% | 43.33% | 45.83% | 5.78 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 778 | 333 | 445 | 42.80% | 40.00% | 43.96% | 7.20 pp | -112 | 42 | -2.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 601 | 297 | 304 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 601 | 283 | 318 | 47.09% | 45.00% | 48.33% | 2.91 pp | -35 | 38 | -0.92 |
| BTC Daily | lstm | LSTM | 601 | 268 | 333 | 44.59% | 42.92% | 44.79% | 5.41 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 601 | 261 | 340 | 43.43% | 45.00% | 44.17% | 6.57 pp | -79 | 38 | -2.08 |
| BTC Daily | xgb | XGBoost | 611 | 246 | 365 | 40.26% | 35.83% | 40.42% | 9.74 pp | -119 | 38 | -3.13 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 427 | 195 | 232 | 45.67% | 45.83% | 45.67% | 4.33 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 427 | 194 | 233 | 45.43% | 46.67% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 427 | 194 | 233 | 45.43% | 47.92% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 427 | 173 | 254 | 40.52% | 40.42% | 40.52% | 9.48 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 36 | 21 | 15 | 58.33% | 58.33% | 58.33% | 8.33 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 36 | 14 | 22 | 38.89% | 38.89% | 38.89% | 11.11 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 21 | 15 | 58.33% | 58.33% | 58.33% | 8.33 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 14 | 22 | 38.89% | 38.89% | 38.89% | 11.11 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
