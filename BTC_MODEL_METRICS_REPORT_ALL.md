# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T08:47:49.701179+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1160 | 872 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1036 | 671 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 652 | 433 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 653 | 486 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 85 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 12:00:00+00:00 | 85 | 2 | 83 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 661 | 345 | 316 | 52.19% | 50.42% | 53.33% | 2.19 pp | 29 | 40 | 0.72 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 433 | 220 | 213 | 50.81% | 47.08% | 50.81% | 0.81 pp | 7 | 43 | 0.16 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | nn | NN | 433 | 213 | 220 | 49.19% | 50.00% | 49.19% | 0.81 pp | -7 | 43 | -0.16 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 838 | 415 | 423 | 49.52% | 48.33% | 49.38% | 0.48 pp | -8 | 45 | -0.18 |
| BTC Daily | nn | NN | 661 | 320 | 341 | 48.41% | 47.08% | 50.42% | 1.59 pp | -21 | 40 | -0.53 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | nn | NN | 486 | 228 | 258 | 46.91% | 44.58% | 47.29% | 3.09 pp | -30 | 43 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| BTC Daily | transformer | Transformer | 661 | 312 | 349 | 47.20% | 44.17% | 48.96% | 2.80 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 486 | 223 | 263 | 45.88% | 47.08% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 486 | 223 | 263 | 45.88% | 43.33% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Hourly | nn | NN | 838 | 398 | 440 | 47.49% | 47.50% | 46.88% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | transformer | Transformer | 838 | 398 | 440 | 47.49% | 47.08% | 45.21% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Market Hours | transformer | Transformer | 433 | 196 | 237 | 45.27% | 40.83% | 45.27% | 4.73 pp | -41 | 43 | -0.95 |
| BTC Hourly | rf | RandomForest | 838 | 393 | 445 | 46.90% | 46.25% | 45.83% | 3.10 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 433 | 190 | 243 | 43.88% | 44.58% | 43.88% | 6.12 pp | -53 | 43 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 486 | 214 | 272 | 44.03% | 43.33% | 44.17% | 5.97 pp | -58 | 43 | -1.35 |
| BTC Daily | lstm | LSTM | 661 | 303 | 358 | 45.84% | 40.00% | 45.62% | 4.16 pp | -55 | 40 | -1.38 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |
| BTC Market Hours | lstm | LSTM | 433 | 185 | 248 | 42.73% | 40.83% | 42.73% | 7.27 pp | -63 | 43 | -1.47 |
| BTC Daily | rf | RandomForest | 661 | 299 | 362 | 45.23% | 42.92% | 46.25% | 4.77 pp | -63 | 40 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 486 | 202 | 284 | 41.56% | 40.00% | 41.67% | 8.44 pp | -82 | 43 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 433 | 175 | 258 | 40.42% | 38.75% | 40.42% | 9.58 pp | -83 | 43 | -1.93 |
| BTC Hourly | lstm | LSTM | 838 | 374 | 464 | 44.63% | 43.75% | 45.21% | 5.37 pp | -90 | 45 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | xgb | XGBoost | 838 | 369 | 469 | 44.03% | 43.75% | 43.75% | 5.97 pp | -100 | 45 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 486 | 194 | 292 | 39.92% | 36.25% | 40.00% | 10.08 pp | -98 | 43 | -2.28 |
| BTC Daily | xgb | XGBoost | 671 | 271 | 400 | 40.39% | 35.00% | 40.83% | 9.61 pp | -129 | 40 | -3.23 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 838 | 415 | 423 | 49.52% | 48.33% | 49.38% | 0.48 pp | -8 | 45 | -0.18 |
| BTC Hourly | nn | NN | 838 | 398 | 440 | 47.49% | 47.50% | 46.88% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | transformer | Transformer | 838 | 398 | 440 | 47.49% | 47.08% | 45.21% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | rf | RandomForest | 838 | 393 | 445 | 46.90% | 46.25% | 45.83% | 3.10 pp | -52 | 45 | -1.16 |
| BTC Hourly | lstm | LSTM | 838 | 374 | 464 | 44.63% | 43.75% | 45.21% | 5.37 pp | -90 | 45 | -2.00 |
| BTC Hourly | xgb | XGBoost | 838 | 369 | 469 | 44.03% | 43.75% | 43.75% | 5.97 pp | -100 | 45 | -2.22 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 661 | 345 | 316 | 52.19% | 50.42% | 53.33% | 2.19 pp | 29 | 40 | 0.72 |
| BTC Daily | nn | NN | 661 | 320 | 341 | 48.41% | 47.08% | 50.42% | 1.59 pp | -21 | 40 | -0.53 |
| BTC Daily | transformer | Transformer | 661 | 312 | 349 | 47.20% | 44.17% | 48.96% | 2.80 pp | -37 | 40 | -0.93 |
| BTC Daily | lstm | LSTM | 661 | 303 | 358 | 45.84% | 40.00% | 45.62% | 4.16 pp | -55 | 40 | -1.38 |
| BTC Daily | rf | RandomForest | 661 | 299 | 362 | 45.23% | 42.92% | 46.25% | 4.77 pp | -63 | 40 | -1.57 |
| BTC Daily | xgb | XGBoost | 671 | 271 | 400 | 40.39% | 35.00% | 40.83% | 9.61 pp | -129 | 40 | -3.23 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 433 | 220 | 213 | 50.81% | 47.08% | 50.81% | 0.81 pp | 7 | 43 | 0.16 |
| BTC Market Hours | nn | NN | 433 | 213 | 220 | 49.19% | 50.00% | 49.19% | 0.81 pp | -7 | 43 | -0.16 |
| BTC Market Hours | transformer | Transformer | 433 | 196 | 237 | 45.27% | 40.83% | 45.27% | 4.73 pp | -41 | 43 | -0.95 |
| BTC Market Hours | rf | RandomForest | 433 | 190 | 243 | 43.88% | 44.58% | 43.88% | 6.12 pp | -53 | 43 | -1.23 |
| BTC Market Hours | lstm | LSTM | 433 | 185 | 248 | 42.73% | 40.83% | 42.73% | 7.27 pp | -63 | 43 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 433 | 175 | 258 | 40.42% | 38.75% | 40.42% | 9.58 pp | -83 | 43 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 486 | 228 | 258 | 46.91% | 44.58% | 47.29% | 3.09 pp | -30 | 43 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 486 | 223 | 263 | 45.88% | 47.08% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 486 | 223 | 263 | 45.88% | 43.33% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 486 | 214 | 272 | 44.03% | 43.33% | 44.17% | 5.97 pp | -58 | 43 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 486 | 202 | 284 | 41.56% | 40.00% | 41.67% | 8.44 pp | -82 | 43 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 486 | 194 | 292 | 39.92% | 36.25% | 40.00% | 10.08 pp | -98 | 43 | -2.28 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 47 | 38 | 55.29% | 55.29% | 55.29% | 5.29 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 43 | 42 | 50.59% | 50.59% | 50.59% | 0.59 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 39 | 46 | 45.88% | 45.88% | 45.88% | 4.12 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 36 | 49 | 42.35% | 42.35% | 42.35% | 7.65 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
