# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T22:29:41.198060+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 1102 | 791 | 311 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 926 | 579 | 346 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 487 | 341 | 145 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 21:00:00+00:00 | 489 | 395 | 92 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 7 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 7 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 0 | 7 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 02:00:00+00:00 | 7 | 0 | 7 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 341 | 169 | 172 | 49.56% | 47.92% | 49.56% | 0.44 pp | -3 | 36 | -0.08 |
| BTC Daily | transformer | Transformer | 569 | 279 | 290 | 49.03% | 52.92% | 48.75% | 0.97 pp | -11 | 37 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 569 | 278 | 291 | 48.86% | 47.92% | 49.17% | 1.14 pp | -13 | 37 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 341 | 160 | 181 | 46.92% | 46.25% | 46.92% | 3.08 pp | -21 | 36 | -0.58 |
| BTC Daily | nn | NN | 569 | 270 | 299 | 47.45% | 46.67% | 48.33% | 2.55 pp | -29 | 37 | -0.78 |
| BTC Market Hours Daily | nn | NN | 395 | 183 | 212 | 46.33% | 48.33% | 46.33% | 3.67 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 395 | 180 | 215 | 45.57% | 45.42% | 45.57% | 4.43 pp | -35 | 36 | -0.97 |
| BTC Market Hours | nn | NN | 341 | 153 | 188 | 44.87% | 46.67% | 44.87% | 5.13 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 395 | 178 | 217 | 45.06% | 44.58% | 45.06% | 4.94 pp | -39 | 36 | -1.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 341 | 149 | 192 | 43.70% | 44.58% | 43.70% | 6.30 pp | -43 | 36 | -1.19 |
| BTC Market Hours | rf | RandomForest | 341 | 144 | 197 | 42.23% | 42.50% | 42.23% | 7.77 pp | -53 | 36 | -1.47 |
| BTC Daily | lstm | LSTM | 569 | 257 | 312 | 45.17% | 46.25% | 45.00% | 4.83 pp | -55 | 37 | -1.49 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 341 | 140 | 201 | 41.06% | 40.83% | 41.06% | 8.94 pp | -61 | 36 | -1.69 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 569 | 248 | 321 | 43.59% | 46.25% | 44.17% | 6.41 pp | -73 | 37 | -1.97 |
| BTC Market Hours Daily | rf | RandomForest | 395 | 161 | 234 | 40.76% | 38.33% | 40.76% | 9.24 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 395 | 160 | 235 | 40.51% | 39.17% | 40.51% | 9.49 pp | -75 | 36 | -2.08 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 395 | 156 | 239 | 39.49% | 37.08% | 39.49% | 10.51 pp | -83 | 36 | -2.31 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| Consolidated Hourly | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 579 | 236 | 343 | 40.76% | 36.67% | 40.83% | 9.24 pp | -107 | 37 | -2.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 569 | 279 | 290 | 49.03% | 52.92% | 48.75% | 0.97 pp | -11 | 37 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 569 | 278 | 291 | 48.86% | 47.92% | 49.17% | 1.14 pp | -13 | 37 | -0.35 |
| BTC Daily | nn | NN | 569 | 270 | 299 | 47.45% | 46.67% | 48.33% | 2.55 pp | -29 | 37 | -0.78 |
| BTC Daily | lstm | LSTM | 569 | 257 | 312 | 45.17% | 46.25% | 45.00% | 4.83 pp | -55 | 37 | -1.49 |
| BTC Daily | rf | RandomForest | 569 | 248 | 321 | 43.59% | 46.25% | 44.17% | 6.41 pp | -73 | 37 | -1.97 |
| BTC Daily | xgb | XGBoost | 579 | 236 | 343 | 40.76% | 36.67% | 40.83% | 9.24 pp | -107 | 37 | -2.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 341 | 169 | 172 | 49.56% | 47.92% | 49.56% | 0.44 pp | -3 | 36 | -0.08 |
| BTC Market Hours | transformer | Transformer | 341 | 160 | 181 | 46.92% | 46.25% | 46.92% | 3.08 pp | -21 | 36 | -0.58 |
| BTC Market Hours | nn | NN | 341 | 153 | 188 | 44.87% | 46.67% | 44.87% | 5.13 pp | -35 | 36 | -0.97 |
| BTC Market Hours | lstm | LSTM | 341 | 149 | 192 | 43.70% | 44.58% | 43.70% | 6.30 pp | -43 | 36 | -1.19 |
| BTC Market Hours | rf | RandomForest | 341 | 144 | 197 | 42.23% | 42.50% | 42.23% | 7.77 pp | -53 | 36 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 341 | 140 | 201 | 41.06% | 40.83% | 41.06% | 8.94 pp | -61 | 36 | -1.69 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 395 | 183 | 212 | 46.33% | 48.33% | 46.33% | 3.67 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 395 | 180 | 215 | 45.57% | 45.42% | 45.57% | 4.43 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 395 | 178 | 217 | 45.06% | 44.58% | 45.06% | 4.94 pp | -39 | 36 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 395 | 161 | 234 | 40.76% | 38.33% | 40.76% | 9.24 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 395 | 160 | 235 | 40.51% | 39.17% | 40.51% | 9.49 pp | -75 | 36 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 395 | 156 | 239 | 39.49% | 37.08% | 39.49% | 10.51 pp | -83 | 36 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 2 | -2.50 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
