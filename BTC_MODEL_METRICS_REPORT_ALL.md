# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T16:59:19.783396+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 805 | 314 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 969 | 604 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 537 | 366 | 170 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 15:00:00+00:00 | 538 | 419 | 117 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 594 | 294 | 300 | 49.49% | 51.25% | 50.42% | 0.51 pp | -6 | 38 | -0.16 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 366 | 180 | 186 | 49.18% | 48.75% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 594 | 290 | 304 | 48.82% | 46.67% | 49.17% | 1.18 pp | -14 | 38 | -0.37 |
| BTC Market Hours | transformer | Transformer | 366 | 174 | 192 | 47.54% | 46.67% | 47.54% | 2.46 pp | -18 | 38 | -0.47 |
| BTC Market Hours | nn | NN | 366 | 168 | 198 | 45.90% | 47.92% | 45.90% | 4.10 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 594 | 280 | 314 | 47.14% | 45.42% | 48.33% | 2.86 pp | -34 | 38 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 419 | 192 | 227 | 45.82% | 45.83% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | nn | NN | 419 | 192 | 227 | 45.82% | 47.08% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 419 | 191 | 228 | 45.58% | 47.08% | 45.58% | 4.42 pp | -37 | 37 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 771 | 363 | 408 | 47.08% | 43.33% | 47.29% | 2.92 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 771 | 361 | 410 | 46.82% | 42.92% | 45.62% | 3.18 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 366 | 157 | 209 | 42.90% | 43.33% | 42.90% | 7.10 pp | -52 | 38 | -1.37 |
| BTC Market Hours | rf | RandomForest | 366 | 156 | 210 | 42.62% | 42.08% | 42.62% | 7.38 pp | -54 | 38 | -1.42 |
| BTC Daily | lstm | LSTM | 594 | 266 | 328 | 44.78% | 44.17% | 45.00% | 5.22 pp | -62 | 38 | -1.63 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 366 | 149 | 217 | 40.71% | 41.67% | 40.71% | 9.29 pp | -68 | 38 | -1.79 |
| BTC Hourly | rf | RandomForest | 771 | 346 | 425 | 44.88% | 45.00% | 44.58% | 5.12 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 771 | 344 | 427 | 44.62% | 39.58% | 45.42% | 5.38 pp | -83 | 42 | -1.98 |
| BTC Daily | rf | RandomForest | 594 | 257 | 337 | 43.27% | 44.58% | 43.75% | 6.73 pp | -80 | 38 | -2.11 |
| BTC Market Hours Daily | rf | RandomForest | 419 | 170 | 249 | 40.57% | 40.00% | 40.57% | 9.43 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 771 | 340 | 431 | 44.10% | 42.92% | 45.42% | 5.90 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 38.75% | 39.86% | 10.14 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 419 | 166 | 253 | 39.62% | 37.92% | 39.62% | 10.38 pp | -87 | 37 | -2.35 |
| BTC Hourly | xgb | XGBoost | 771 | 331 | 440 | 42.93% | 41.25% | 44.38% | 7.07 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 36.25% | 40.00% | 9.77 pp | -118 | 38 | -3.11 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 771 | 363 | 408 | 47.08% | 43.33% | 47.29% | 2.92 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 771 | 361 | 410 | 46.82% | 42.92% | 45.62% | 3.18 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 771 | 346 | 425 | 44.88% | 45.00% | 44.58% | 5.12 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 771 | 344 | 427 | 44.62% | 39.58% | 45.42% | 5.38 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 771 | 340 | 431 | 44.10% | 42.92% | 45.42% | 5.90 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 771 | 331 | 440 | 42.93% | 41.25% | 44.38% | 7.07 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 594 | 294 | 300 | 49.49% | 51.25% | 50.42% | 0.51 pp | -6 | 38 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 594 | 290 | 304 | 48.82% | 46.67% | 49.17% | 1.18 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 594 | 280 | 314 | 47.14% | 45.42% | 48.33% | 2.86 pp | -34 | 38 | -0.89 |
| BTC Daily | lstm | LSTM | 594 | 266 | 328 | 44.78% | 44.17% | 45.00% | 5.22 pp | -62 | 38 | -1.63 |
| BTC Daily | rf | RandomForest | 594 | 257 | 337 | 43.27% | 44.58% | 43.75% | 6.73 pp | -80 | 38 | -2.11 |
| BTC Daily | xgb | XGBoost | 604 | 243 | 361 | 40.23% | 36.25% | 40.00% | 9.77 pp | -118 | 38 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 366 | 180 | 186 | 49.18% | 48.75% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 366 | 174 | 192 | 47.54% | 46.67% | 47.54% | 2.46 pp | -18 | 38 | -0.47 |
| BTC Market Hours | nn | NN | 366 | 168 | 198 | 45.90% | 47.92% | 45.90% | 4.10 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 366 | 157 | 209 | 42.90% | 43.33% | 42.90% | 7.10 pp | -52 | 38 | -1.37 |
| BTC Market Hours | rf | RandomForest | 366 | 156 | 210 | 42.62% | 42.08% | 42.62% | 7.38 pp | -54 | 38 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 366 | 149 | 217 | 40.71% | 41.67% | 40.71% | 9.29 pp | -68 | 38 | -1.79 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 419 | 192 | 227 | 45.82% | 45.83% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | nn | NN | 419 | 192 | 227 | 45.82% | 47.08% | 45.82% | 4.18 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 419 | 191 | 228 | 45.58% | 47.08% | 45.58% | 4.42 pp | -37 | 37 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 419 | 170 | 249 | 40.57% | 40.00% | 40.57% | 9.43 pp | -79 | 37 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 419 | 167 | 252 | 39.86% | 38.75% | 39.86% | 10.14 pp | -85 | 37 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 419 | 166 | 253 | 39.62% | 37.92% | 39.62% | 10.38 pp | -87 | 37 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

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
