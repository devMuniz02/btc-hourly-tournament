# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T14:20:08.551852+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 821 | 298 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 985 | 620 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 13:00:00+00:00 | 564 | 382 | 181 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 13:00:00+00:00 | 566 | 436 | 128 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T04:00:00+00:00 | 42 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T04:00:00+00:00 | 42 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T04:00:00+00:00 | 42 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T04:00:00+00:00 | 43 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 382 | 187 | 195 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 610 | 299 | 311 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 610 | 299 | 311 | 49.02% | 49.58% | 50.00% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Market Hours | transformer | Transformer | 382 | 178 | 204 | 46.60% | 43.75% | 46.60% | 3.40 pp | -26 | 39 | -0.67 |
| BTC Market Hours | nn | NN | 382 | 177 | 205 | 46.34% | 48.75% | 46.34% | 3.66 pp | -28 | 39 | -0.72 |
| Consolidated Hourly | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 610 | 287 | 323 | 47.05% | 44.17% | 48.54% | 2.95 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 436 | 199 | 237 | 45.64% | 47.08% | 45.64% | 4.36 pp | -38 | 39 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 787 | 372 | 415 | 47.27% | 44.58% | 47.29% | 2.73 pp | -43 | 43 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 436 | 198 | 238 | 45.41% | 45.42% | 45.41% | 4.59 pp | -40 | 39 | -1.03 |
| BTC Market Hours Daily | nn | NN | 436 | 197 | 239 | 45.18% | 45.83% | 45.18% | 4.82 pp | -42 | 39 | -1.08 |
| BTC Hourly | transformer | Transformer | 787 | 369 | 418 | 46.89% | 43.33% | 46.04% | 3.11 pp | -49 | 43 | -1.14 |
| Consolidated Hourly | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 382 | 163 | 219 | 42.67% | 43.33% | 42.67% | 7.33 pp | -56 | 39 | -1.44 |
| BTC Market Hours | rf | RandomForest | 382 | 162 | 220 | 42.41% | 40.42% | 42.41% | 7.59 pp | -58 | 39 | -1.49 |
| BTC Daily | lstm | LSTM | 610 | 272 | 338 | 44.59% | 43.75% | 44.58% | 5.41 pp | -66 | 38 | -1.74 |
| BTC Hourly | nn | NN | 787 | 354 | 433 | 44.98% | 40.42% | 45.83% | 5.02 pp | -79 | 43 | -1.84 |
| BTC Market Hours | xgb | XGBoost | 382 | 153 | 229 | 40.05% | 38.33% | 40.05% | 9.95 pp | -76 | 39 | -1.95 |
| BTC Hourly | rf | RandomForest | 787 | 350 | 437 | 44.47% | 42.92% | 43.96% | 5.53 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 436 | 176 | 260 | 40.37% | 38.75% | 40.37% | 9.63 pp | -84 | 39 | -2.15 |
| BTC Hourly | lstm | LSTM | 787 | 347 | 440 | 44.09% | 43.75% | 45.42% | 5.91 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 610 | 263 | 347 | 43.11% | 43.33% | 43.75% | 6.89 pp | -84 | 38 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 436 | 172 | 264 | 39.45% | 37.50% | 39.45% | 10.55 pp | -92 | 39 | -2.36 |
| Consolidated Hourly | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 436 | 171 | 265 | 39.22% | 37.92% | 39.22% | 10.78 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 787 | 336 | 451 | 42.69% | 40.00% | 44.17% | 7.31 pp | -115 | 43 | -2.67 |
| BTC Daily | xgb | XGBoost | 620 | 247 | 373 | 39.84% | 33.75% | 40.00% | 10.16 pp | -126 | 38 | -3.32 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 787 | 372 | 415 | 47.27% | 44.58% | 47.29% | 2.73 pp | -43 | 43 | -1.00 |
| BTC Hourly | transformer | Transformer | 787 | 369 | 418 | 46.89% | 43.33% | 46.04% | 3.11 pp | -49 | 43 | -1.14 |
| BTC Hourly | nn | NN | 787 | 354 | 433 | 44.98% | 40.42% | 45.83% | 5.02 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 787 | 350 | 437 | 44.47% | 42.92% | 43.96% | 5.53 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 787 | 347 | 440 | 44.09% | 43.75% | 45.42% | 5.91 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 787 | 336 | 451 | 42.69% | 40.00% | 44.17% | 7.31 pp | -115 | 43 | -2.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 610 | 299 | 311 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | transformer | Transformer | 610 | 299 | 311 | 49.02% | 49.58% | 50.00% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 610 | 287 | 323 | 47.05% | 44.17% | 48.54% | 2.95 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 610 | 272 | 338 | 44.59% | 43.75% | 44.58% | 5.41 pp | -66 | 38 | -1.74 |
| BTC Daily | rf | RandomForest | 610 | 263 | 347 | 43.11% | 43.33% | 43.75% | 6.89 pp | -84 | 38 | -2.21 |
| BTC Daily | xgb | XGBoost | 620 | 247 | 373 | 39.84% | 33.75% | 40.00% | 10.16 pp | -126 | 38 | -3.32 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 382 | 187 | 195 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Market Hours | transformer | Transformer | 382 | 178 | 204 | 46.60% | 43.75% | 46.60% | 3.40 pp | -26 | 39 | -0.67 |
| BTC Market Hours | nn | NN | 382 | 177 | 205 | 46.34% | 48.75% | 46.34% | 3.66 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 382 | 163 | 219 | 42.67% | 43.33% | 42.67% | 7.33 pp | -56 | 39 | -1.44 |
| BTC Market Hours | rf | RandomForest | 382 | 162 | 220 | 42.41% | 40.42% | 42.41% | 7.59 pp | -58 | 39 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 382 | 153 | 229 | 40.05% | 38.33% | 40.05% | 9.95 pp | -76 | 39 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 436 | 199 | 237 | 45.64% | 47.08% | 45.64% | 4.36 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 436 | 198 | 238 | 45.41% | 45.42% | 45.41% | 4.59 pp | -40 | 39 | -1.03 |
| BTC Market Hours Daily | nn | NN | 436 | 197 | 239 | 45.18% | 45.83% | 45.18% | 4.82 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 436 | 176 | 260 | 40.37% | 38.75% | 40.37% | 9.63 pp | -84 | 39 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 436 | 172 | 264 | 39.45% | 37.50% | 39.45% | 10.55 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 436 | 171 | 265 | 39.22% | 37.92% | 39.22% | 10.78 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Hourly | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 42 | 25 | 17 | 59.52% | 59.52% | 59.52% | 9.52 pp | 8 | 5 | 1.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 42 | 23 | 19 | 54.76% | 54.76% | 54.76% | 4.76 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
