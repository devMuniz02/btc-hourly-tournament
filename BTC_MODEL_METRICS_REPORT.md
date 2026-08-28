# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T11:06:42.969152+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 818 | 301 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 982 | 617 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 559 | 379 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 561 | 433 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 12:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 12:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 12:00:00+00:00 | 40 | 1 | 39 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 12:00:00+00:00 | 40 | 1 | 39 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 379 | 186 | 193 | 49.08% | 47.92% | 49.08% | 0.92 pp | -7 | 39 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 607 | 298 | 309 | 49.09% | 48.33% | 50.21% | 0.91 pp | -11 | 38 | -0.29 |
| BTC Daily | transformer | Transformer | 607 | 298 | 309 | 49.09% | 50.00% | 49.79% | 0.91 pp | -11 | 38 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 379 | 178 | 201 | 46.97% | 44.58% | 46.97% | 3.03 pp | -23 | 39 | -0.59 |
| BTC Market Hours | nn | NN | 379 | 175 | 204 | 46.17% | 48.75% | 46.17% | 3.83 pp | -29 | 39 | -0.74 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 784 | 372 | 412 | 47.45% | 45.42% | 47.92% | 2.55 pp | -40 | 43 | -0.93 |
| BTC Daily | nn | NN | 607 | 285 | 322 | 46.95% | 44.17% | 48.12% | 3.05 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 433 | 198 | 235 | 45.73% | 45.83% | 45.73% | 4.27 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 433 | 197 | 236 | 45.50% | 46.67% | 45.50% | 4.50 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 433 | 197 | 236 | 45.50% | 47.08% | 45.50% | 4.50 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 784 | 368 | 416 | 46.94% | 43.33% | 46.04% | 3.06 pp | -48 | 43 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 379 | 162 | 217 | 42.74% | 43.75% | 42.74% | 7.26 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 379 | 160 | 219 | 42.22% | 40.42% | 42.22% | 7.78 pp | -59 | 39 | -1.51 |
| BTC Daily | lstm | LSTM | 607 | 270 | 337 | 44.48% | 43.33% | 44.17% | 5.52 pp | -67 | 38 | -1.76 |
| BTC Hourly | nn | NN | 784 | 353 | 431 | 45.03% | 40.42% | 45.83% | 4.97 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 784 | 350 | 434 | 44.64% | 43.33% | 44.38% | 5.36 pp | -84 | 43 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 379 | 151 | 228 | 39.84% | 38.75% | 39.84% | 10.16 pp | -77 | 39 | -1.97 |
| BTC Hourly | lstm | LSTM | 784 | 347 | 437 | 44.26% | 43.75% | 45.62% | 5.74 pp | -90 | 43 | -2.09 |
| BTC Daily | rf | RandomForest | 607 | 262 | 345 | 43.16% | 43.75% | 43.54% | 6.84 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | rf | RandomForest | 433 | 175 | 258 | 40.42% | 39.17% | 40.42% | 9.58 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 433 | 171 | 262 | 39.49% | 37.92% | 39.49% | 10.51 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 433 | 170 | 263 | 39.26% | 38.33% | 39.26% | 10.74 pp | -93 | 38 | -2.45 |
| BTC Hourly | xgb | XGBoost | 784 | 336 | 448 | 42.86% | 40.42% | 44.38% | 7.14 pp | -112 | 43 | -2.60 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 617 | 247 | 370 | 40.03% | 34.17% | 40.21% | 9.97 pp | -123 | 38 | -3.24 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 784 | 372 | 412 | 47.45% | 45.42% | 47.92% | 2.55 pp | -40 | 43 | -0.93 |
| BTC Hourly | transformer | Transformer | 784 | 368 | 416 | 46.94% | 43.33% | 46.04% | 3.06 pp | -48 | 43 | -1.12 |
| BTC Hourly | nn | NN | 784 | 353 | 431 | 45.03% | 40.42% | 45.83% | 4.97 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 784 | 350 | 434 | 44.64% | 43.33% | 44.38% | 5.36 pp | -84 | 43 | -1.95 |
| BTC Hourly | lstm | LSTM | 784 | 347 | 437 | 44.26% | 43.75% | 45.62% | 5.74 pp | -90 | 43 | -2.09 |
| BTC Hourly | xgb | XGBoost | 784 | 336 | 448 | 42.86% | 40.42% | 44.38% | 7.14 pp | -112 | 43 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 607 | 298 | 309 | 49.09% | 48.33% | 50.21% | 0.91 pp | -11 | 38 | -0.29 |
| BTC Daily | transformer | Transformer | 607 | 298 | 309 | 49.09% | 50.00% | 49.79% | 0.91 pp | -11 | 38 | -0.29 |
| BTC Daily | nn | NN | 607 | 285 | 322 | 46.95% | 44.17% | 48.12% | 3.05 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 607 | 270 | 337 | 44.48% | 43.33% | 44.17% | 5.52 pp | -67 | 38 | -1.76 |
| BTC Daily | rf | RandomForest | 607 | 262 | 345 | 43.16% | 43.75% | 43.54% | 6.84 pp | -83 | 38 | -2.18 |
| BTC Daily | xgb | XGBoost | 617 | 247 | 370 | 40.03% | 34.17% | 40.21% | 9.97 pp | -123 | 38 | -3.24 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 379 | 186 | 193 | 49.08% | 47.92% | 49.08% | 0.92 pp | -7 | 39 | -0.18 |
| BTC Market Hours | transformer | Transformer | 379 | 178 | 201 | 46.97% | 44.58% | 46.97% | 3.03 pp | -23 | 39 | -0.59 |
| BTC Market Hours | nn | NN | 379 | 175 | 204 | 46.17% | 48.75% | 46.17% | 3.83 pp | -29 | 39 | -0.74 |
| BTC Market Hours | lstm | LSTM | 379 | 162 | 217 | 42.74% | 43.75% | 42.74% | 7.26 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 379 | 160 | 219 | 42.22% | 40.42% | 42.22% | 7.78 pp | -59 | 39 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 379 | 151 | 228 | 39.84% | 38.75% | 39.84% | 10.16 pp | -77 | 39 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 433 | 198 | 235 | 45.73% | 45.83% | 45.73% | 4.27 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 433 | 197 | 236 | 45.50% | 46.67% | 45.50% | 4.50 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 433 | 197 | 236 | 45.50% | 47.08% | 45.50% | 4.50 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 433 | 175 | 258 | 40.42% | 39.17% | 40.42% | 9.58 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 433 | 171 | 262 | 39.49% | 37.92% | 39.49% | 10.51 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 433 | 170 | 263 | 39.26% | 38.33% | 39.26% | 10.74 pp | -93 | 38 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
