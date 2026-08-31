# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T09:08:42.495099+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 654 | 487 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 661 | 345 | 316 | 52.19% | 50.42% | 53.33% | 2.19 pp | 29 | 40 | 0.72 |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 433 | 220 | 213 | 50.81% | 47.08% | 50.81% | 0.81 pp | 7 | 43 | 0.16 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | nn | NN | 433 | 213 | 220 | 49.19% | 50.00% | 49.19% | 0.81 pp | -7 | 43 | -0.16 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 838 | 415 | 423 | 49.52% | 48.33% | 49.38% | 0.48 pp | -8 | 45 | -0.18 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| BTC Daily | nn | NN | 661 | 320 | 341 | 48.41% | 47.08% | 50.42% | 1.59 pp | -21 | 40 | -0.53 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | nn | NN | 487 | 228 | 259 | 46.82% | 44.17% | 47.08% | 3.18 pp | -31 | 43 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 224 | 263 | 46.00% | 43.75% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Daily | transformer | Transformer | 661 | 312 | 349 | 47.20% | 44.17% | 48.96% | 2.80 pp | -37 | 40 | -0.93 |
| BTC Hourly | nn | NN | 838 | 398 | 440 | 47.49% | 47.50% | 46.88% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Hourly | transformer | Transformer | 838 | 398 | 440 | 47.49% | 47.08% | 45.21% | 2.51 pp | -42 | 45 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 223 | 264 | 45.79% | 47.08% | 45.83% | 4.21 pp | -41 | 43 | -0.95 |
| BTC Market Hours | transformer | Transformer | 433 | 196 | 237 | 45.27% | 40.83% | 45.27% | 4.73 pp | -41 | 43 | -0.95 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 838 | 393 | 445 | 46.90% | 46.25% | 45.83% | 3.10 pp | -52 | 45 | -1.16 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| BTC Market Hours | rf | RandomForest | 433 | 190 | 243 | 43.88% | 44.58% | 43.88% | 6.12 pp | -53 | 43 | -1.23 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 214 | 273 | 43.94% | 43.33% | 44.17% | 6.06 pp | -59 | 43 | -1.37 |
| BTC Daily | lstm | LSTM | 661 | 303 | 358 | 45.84% | 40.00% | 45.62% | 4.16 pp | -55 | 40 | -1.38 |
| BTC Market Hours | lstm | LSTM | 433 | 185 | 248 | 42.73% | 40.83% | 42.73% | 7.27 pp | -63 | 43 | -1.47 |
| BTC Daily | rf | RandomForest | 661 | 299 | 362 | 45.23% | 42.92% | 46.25% | 4.77 pp | -63 | 40 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 202 | 285 | 41.48% | 39.58% | 41.67% | 8.52 pp | -83 | 43 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 433 | 175 | 258 | 40.42% | 38.75% | 40.42% | 9.58 pp | -83 | 43 | -1.93 |
| BTC Hourly | lstm | LSTM | 838 | 374 | 464 | 44.63% | 43.75% | 45.21% | 5.37 pp | -90 | 45 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | xgb | XGBoost | 838 | 369 | 469 | 44.03% | 43.75% | 43.75% | 5.97 pp | -100 | 45 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 195 | 292 | 40.04% | 36.67% | 40.21% | 9.96 pp | -97 | 43 | -2.26 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
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
| BTC Market Hours Daily | nn | NN | 487 | 228 | 259 | 46.82% | 44.17% | 47.08% | 3.18 pp | -31 | 43 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 487 | 224 | 263 | 46.00% | 43.75% | 46.04% | 4.00 pp | -39 | 43 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 487 | 223 | 264 | 45.79% | 47.08% | 45.83% | 4.21 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 487 | 214 | 273 | 43.94% | 43.33% | 44.17% | 6.06 pp | -59 | 43 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 487 | 202 | 285 | 41.48% | 39.58% | 41.67% | 8.52 pp | -83 | 43 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 487 | 195 | 292 | 40.04% | 36.67% | 40.21% | 9.96 pp | -97 | 43 | -2.26 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

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
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
