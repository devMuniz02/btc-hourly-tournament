# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T10:57:46.271214+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1125 | 837 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1001 | 636 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 591 | 398 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 593 | 452 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 56 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T07:00:00+00:00 | 57 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 398 | 196 | 202 | 49.25% | 47.50% | 49.25% | 0.75 pp | -6 | 40 | -0.15 |
| BTC Daily | transformer | Transformer | 626 | 308 | 318 | 49.20% | 47.92% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 626 | 306 | 320 | 48.88% | 47.08% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 398 | 187 | 211 | 46.98% | 49.58% | 46.98% | 3.02 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 398 | 185 | 213 | 46.48% | 42.50% | 46.48% | 3.52 pp | -28 | 40 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 452 | 209 | 243 | 46.24% | 47.92% | 46.24% | 3.76 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 452 | 207 | 245 | 45.80% | 45.00% | 45.80% | 4.20 pp | -38 | 40 | -0.95 |
| BTC Daily | nn | NN | 626 | 294 | 332 | 46.96% | 43.75% | 48.96% | 3.04 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 452 | 206 | 246 | 45.58% | 46.25% | 45.58% | 4.42 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 803 | 379 | 424 | 47.20% | 44.58% | 46.88% | 2.80 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 803 | 378 | 425 | 47.07% | 45.00% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Market Hours | lstm | LSTM | 398 | 174 | 224 | 43.72% | 44.17% | 43.72% | 6.28 pp | -50 | 40 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| BTC Market Hours | rf | RandomForest | 398 | 169 | 229 | 42.46% | 40.83% | 42.46% | 7.54 pp | -60 | 40 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 626 | 278 | 348 | 44.41% | 42.50% | 44.17% | 5.59 pp | -70 | 39 | -1.79 |
| BTC Hourly | nn | NN | 803 | 361 | 442 | 44.96% | 40.83% | 45.00% | 5.04 pp | -81 | 43 | -1.88 |
| BTC Market Hours | xgb | XGBoost | 398 | 161 | 237 | 40.45% | 38.33% | 40.45% | 9.55 pp | -76 | 40 | -1.90 |
| BTC Hourly | rf | RandomForest | 803 | 358 | 445 | 44.58% | 44.17% | 44.17% | 5.42 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 452 | 184 | 268 | 40.71% | 40.00% | 40.71% | 9.29 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 452 | 182 | 270 | 40.27% | 38.75% | 40.27% | 9.73 pp | -88 | 40 | -2.20 |
| BTC Hourly | lstm | LSTM | 803 | 352 | 451 | 43.84% | 43.33% | 45.00% | 6.16 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 626 | 268 | 358 | 42.81% | 42.50% | 43.54% | 7.19 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 452 | 178 | 274 | 39.38% | 37.50% | 39.38% | 10.62 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 803 | 341 | 462 | 42.47% | 40.00% | 43.54% | 7.53 pp | -121 | 43 | -2.81 |
| BTC Daily | xgb | XGBoost | 636 | 250 | 386 | 39.31% | 32.08% | 39.58% | 10.69 pp | -136 | 39 | -3.49 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 803 | 379 | 424 | 47.20% | 44.58% | 46.88% | 2.80 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 803 | 378 | 425 | 47.07% | 45.00% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 803 | 361 | 442 | 44.96% | 40.83% | 45.00% | 5.04 pp | -81 | 43 | -1.88 |
| BTC Hourly | rf | RandomForest | 803 | 358 | 445 | 44.58% | 44.17% | 44.17% | 5.42 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 803 | 352 | 451 | 43.84% | 43.33% | 45.00% | 6.16 pp | -99 | 43 | -2.30 |
| BTC Hourly | xgb | XGBoost | 803 | 341 | 462 | 42.47% | 40.00% | 43.54% | 7.53 pp | -121 | 43 | -2.81 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 626 | 308 | 318 | 49.20% | 47.92% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 626 | 306 | 320 | 48.88% | 47.08% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 626 | 294 | 332 | 46.96% | 43.75% | 48.96% | 3.04 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 626 | 278 | 348 | 44.41% | 42.50% | 44.17% | 5.59 pp | -70 | 39 | -1.79 |
| BTC Daily | rf | RandomForest | 626 | 268 | 358 | 42.81% | 42.50% | 43.54% | 7.19 pp | -90 | 39 | -2.31 |
| BTC Daily | xgb | XGBoost | 636 | 250 | 386 | 39.31% | 32.08% | 39.58% | 10.69 pp | -136 | 39 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 398 | 196 | 202 | 49.25% | 47.50% | 49.25% | 0.75 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 398 | 187 | 211 | 46.98% | 49.58% | 46.98% | 3.02 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 398 | 185 | 213 | 46.48% | 42.50% | 46.48% | 3.52 pp | -28 | 40 | -0.70 |
| BTC Market Hours | lstm | LSTM | 398 | 174 | 224 | 43.72% | 44.17% | 43.72% | 6.28 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 398 | 169 | 229 | 42.46% | 40.83% | 42.46% | 7.54 pp | -60 | 40 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 398 | 161 | 237 | 40.45% | 38.33% | 40.45% | 9.55 pp | -76 | 40 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 452 | 209 | 243 | 46.24% | 47.92% | 46.24% | 3.76 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 452 | 207 | 245 | 45.80% | 45.00% | 45.80% | 4.20 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 452 | 206 | 246 | 45.58% | 46.25% | 45.58% | 4.42 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 452 | 184 | 268 | 40.71% | 40.00% | 40.71% | 9.29 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 452 | 182 | 270 | 40.27% | 38.75% | 40.27% | 9.73 pp | -88 | 40 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 452 | 178 | 274 | 39.38% | 37.50% | 39.38% | 10.62 pp | -96 | 40 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 56 | 32 | 24 | 57.14% | 57.14% | 57.14% | 7.14 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 56 | 30 | 26 | 53.57% | 53.57% | 53.57% | 3.57 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 56 | 27 | 29 | 48.21% | 48.21% | 48.21% | 1.79 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 56 | 24 | 32 | 42.86% | 42.86% | 42.86% | 7.14 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 56 | 23 | 33 | 41.07% | 41.07% | 41.07% | 8.93 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 56 | 20 | 36 | 35.71% | 35.71% | 35.71% | 14.29 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

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
