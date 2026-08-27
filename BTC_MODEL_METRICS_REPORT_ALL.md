# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T13:39:43.731846+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 802 | 317 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 966 | 601 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 12:00:00+00:00 | 531 | 363 | 167 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 12:00:00+00:00 | 533 | 417 | 114 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 22:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 22:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 22:00:00+00:00 | 28 | 1 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 22:00:00+00:00 | 28 | 1 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 28 | 18 | 10 | 64.29% | 64.29% | 64.29% | 14.29 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 18 | 10 | 64.29% | 64.29% | 64.29% | 14.29 pp | 8 | 3 | 2.67 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 591 | 293 | 298 | 49.58% | 52.08% | 50.21% | 0.42 pp | -5 | 37 | -0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 363 | 178 | 185 | 49.04% | 47.92% | 49.04% | 0.96 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 591 | 289 | 302 | 48.90% | 47.08% | 49.58% | 1.10 pp | -13 | 37 | -0.35 |
| BTC Market Hours | transformer | Transformer | 363 | 172 | 191 | 47.38% | 46.25% | 47.38% | 2.62 pp | -19 | 37 | -0.51 |
| BTC Daily | nn | NN | 591 | 279 | 312 | 47.21% | 45.42% | 48.33% | 2.79 pp | -33 | 37 | -0.89 |
| BTC Market Hours | nn | NN | 363 | 165 | 198 | 45.45% | 47.08% | 45.45% | 4.55 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 417 | 191 | 226 | 45.80% | 46.67% | 45.80% | 4.20 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 417 | 191 | 226 | 45.80% | 47.92% | 45.80% | 4.20 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 417 | 190 | 227 | 45.56% | 45.42% | 45.56% | 4.44 pp | -37 | 37 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 768 | 362 | 406 | 47.14% | 43.33% | 47.29% | 2.86 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 768 | 359 | 409 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 363 | 156 | 207 | 42.98% | 42.92% | 42.98% | 7.02 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 363 | 154 | 209 | 42.42% | 41.25% | 42.42% | 7.58 pp | -55 | 37 | -1.49 |
| BTC Daily | lstm | LSTM | 591 | 265 | 326 | 44.84% | 44.58% | 44.79% | 5.16 pp | -61 | 37 | -1.65 |
| BTC Market Hours | xgb | XGBoost | 363 | 148 | 215 | 40.77% | 41.25% | 40.77% | 9.23 pp | -67 | 37 | -1.81 |
| BTC Hourly | rf | RandomForest | 768 | 344 | 424 | 44.79% | 44.58% | 44.58% | 5.21 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 768 | 343 | 425 | 44.66% | 40.42% | 45.42% | 5.34 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 417 | 170 | 247 | 40.77% | 40.42% | 40.77% | 9.23 pp | -77 | 37 | -2.08 |
| BTC Daily | rf | RandomForest | 591 | 256 | 335 | 43.32% | 44.58% | 43.96% | 6.68 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 768 | 339 | 429 | 44.14% | 43.75% | 45.42% | 5.86 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 417 | 166 | 251 | 39.81% | 37.92% | 39.81% | 10.19 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 417 | 166 | 251 | 39.81% | 38.75% | 39.81% | 10.19 pp | -85 | 37 | -2.30 |
| BTC Hourly | xgb | XGBoost | 768 | 329 | 439 | 42.84% | 41.25% | 44.38% | 7.16 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 601 | 242 | 359 | 40.27% | 36.67% | 40.42% | 9.73 pp | -117 | 37 | -3.16 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 3 | -4.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 3 | -4.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 768 | 362 | 406 | 47.14% | 43.33% | 47.29% | 2.86 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 768 | 359 | 409 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 768 | 344 | 424 | 44.79% | 44.58% | 44.58% | 5.21 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 768 | 343 | 425 | 44.66% | 40.42% | 45.42% | 5.34 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 768 | 339 | 429 | 44.14% | 43.75% | 45.42% | 5.86 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 768 | 329 | 439 | 42.84% | 41.25% | 44.38% | 7.16 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 591 | 293 | 298 | 49.58% | 52.08% | 50.21% | 0.42 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 591 | 289 | 302 | 48.90% | 47.08% | 49.58% | 1.10 pp | -13 | 37 | -0.35 |
| BTC Daily | nn | NN | 591 | 279 | 312 | 47.21% | 45.42% | 48.33% | 2.79 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 591 | 265 | 326 | 44.84% | 44.58% | 44.79% | 5.16 pp | -61 | 37 | -1.65 |
| BTC Daily | rf | RandomForest | 591 | 256 | 335 | 43.32% | 44.58% | 43.96% | 6.68 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 601 | 242 | 359 | 40.27% | 36.67% | 40.42% | 9.73 pp | -117 | 37 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 363 | 178 | 185 | 49.04% | 47.92% | 49.04% | 0.96 pp | -7 | 37 | -0.19 |
| BTC Market Hours | transformer | Transformer | 363 | 172 | 191 | 47.38% | 46.25% | 47.38% | 2.62 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 363 | 165 | 198 | 45.45% | 47.08% | 45.45% | 4.55 pp | -33 | 37 | -0.89 |
| BTC Market Hours | lstm | LSTM | 363 | 156 | 207 | 42.98% | 42.92% | 42.98% | 7.02 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 363 | 154 | 209 | 42.42% | 41.25% | 42.42% | 7.58 pp | -55 | 37 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 363 | 148 | 215 | 40.77% | 41.25% | 40.77% | 9.23 pp | -67 | 37 | -1.81 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 417 | 191 | 226 | 45.80% | 46.67% | 45.80% | 4.20 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 417 | 191 | 226 | 45.80% | 47.92% | 45.80% | 4.20 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 417 | 190 | 227 | 45.56% | 45.42% | 45.56% | 4.44 pp | -37 | 37 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 417 | 170 | 247 | 40.77% | 40.42% | 40.77% | 9.23 pp | -77 | 37 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 417 | 166 | 251 | 39.81% | 37.92% | 39.81% | 10.19 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 417 | 166 | 251 | 39.81% | 38.75% | 39.81% | 10.19 pp | -85 | 37 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 28 | 18 | 10 | 64.29% | 64.29% | 64.29% | 14.29 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 3 | -4.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 18 | 10 | 64.29% | 64.29% | 64.29% | 14.29 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
