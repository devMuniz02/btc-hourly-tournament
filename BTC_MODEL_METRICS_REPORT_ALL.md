# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T12:57:07.337589+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1179 | 891 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1055 | 690 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 684 | 452 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 686 | 506 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T22:00:00+00:00 | 105 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T22:00:00+00:00 | 105 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T22:00:00+00:00 | 105 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T22:00:00+00:00 | 106 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 9 | 4 | 69.23% | 69.23% | 69.23% | 19.23 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 105 | 53 | 52 | 50.48% | 50.48% | 50.48% | 0.48 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 53 | 52 | 50.48% | 50.48% | 50.48% | 0.48 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 680 | 334 | 346 | 49.12% | 47.92% | 49.79% | 0.88 pp | -12 | 41 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 452 | 219 | 233 | 48.45% | 44.58% | 48.45% | 1.55 pp | -14 | 44 | -0.32 |
| Consolidated Hourly | xgb | XGBoost | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Daily | transformer | Transformer | 680 | 329 | 351 | 48.38% | 46.67% | 49.58% | 1.62 pp | -22 | 41 | -0.54 |
| BTC Market Hours | nn | NN | 452 | 213 | 239 | 47.12% | 48.33% | 47.12% | 2.88 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 452 | 209 | 243 | 46.24% | 40.00% | 46.24% | 3.76 pp | -34 | 44 | -0.77 |
| Consolidated Hourly | lstm | LSTM | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 506 | 232 | 274 | 45.85% | 46.25% | 46.25% | 4.15 pp | -42 | 44 | -0.95 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 506 | 231 | 275 | 45.65% | 43.33% | 46.46% | 4.35 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 680 | 319 | 361 | 46.91% | 43.75% | 49.17% | 3.09 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 506 | 230 | 276 | 45.45% | 46.25% | 45.62% | 4.55 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 857 | 404 | 453 | 47.14% | 45.00% | 46.88% | 2.86 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 857 | 403 | 454 | 47.02% | 46.67% | 46.67% | 2.98 pp | -51 | 46 | -1.11 |
| Consolidated Hourly | transformer | Transformer | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| BTC Market Hours | rf | RandomForest | 452 | 196 | 256 | 43.36% | 43.75% | 43.36% | 6.64 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 452 | 191 | 261 | 42.26% | 39.58% | 42.26% | 7.74 pp | -70 | 44 | -1.59 |
| BTC Hourly | nn | NN | 857 | 386 | 471 | 45.04% | 45.00% | 44.17% | 4.96 pp | -85 | 46 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 506 | 210 | 296 | 41.50% | 41.25% | 41.67% | 8.50 pp | -86 | 44 | -1.95 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 452 | 182 | 270 | 40.27% | 38.33% | 40.27% | 9.73 pp | -88 | 44 | -2.00 |
| BTC Hourly | rf | RandomForest | 857 | 382 | 475 | 44.57% | 43.33% | 43.96% | 5.43 pp | -93 | 46 | -2.02 |
| BTC Daily | lstm | LSTM | 680 | 296 | 384 | 43.53% | 38.75% | 42.50% | 6.47 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 506 | 202 | 304 | 39.92% | 37.50% | 40.62% | 10.08 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 680 | 292 | 388 | 42.94% | 40.83% | 43.54% | 7.06 pp | -96 | 41 | -2.34 |
| BTC Market Hours Daily | xgb | XGBoost | 506 | 198 | 308 | 39.13% | 35.83% | 38.75% | 10.87 pp | -110 | 44 | -2.50 |
| BTC Hourly | lstm | LSTM | 857 | 364 | 493 | 42.47% | 37.50% | 41.67% | 7.53 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 857 | 361 | 496 | 42.12% | 40.00% | 42.29% | 7.88 pp | -135 | 46 | -2.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 690 | 274 | 416 | 39.71% | 35.00% | 39.58% | 10.29 pp | -142 | 41 | -3.46 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 857 | 404 | 453 | 47.14% | 45.00% | 46.88% | 2.86 pp | -49 | 46 | -1.07 |
| BTC Hourly | transformer | Transformer | 857 | 403 | 454 | 47.02% | 46.67% | 46.67% | 2.98 pp | -51 | 46 | -1.11 |
| BTC Hourly | nn | NN | 857 | 386 | 471 | 45.04% | 45.00% | 44.17% | 4.96 pp | -85 | 46 | -1.85 |
| BTC Hourly | rf | RandomForest | 857 | 382 | 475 | 44.57% | 43.33% | 43.96% | 5.43 pp | -93 | 46 | -2.02 |
| BTC Hourly | lstm | LSTM | 857 | 364 | 493 | 42.47% | 37.50% | 41.67% | 7.53 pp | -129 | 46 | -2.80 |
| BTC Hourly | xgb | XGBoost | 857 | 361 | 496 | 42.12% | 40.00% | 42.29% | 7.88 pp | -135 | 46 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 680 | 334 | 346 | 49.12% | 47.92% | 49.79% | 0.88 pp | -12 | 41 | -0.29 |
| BTC Daily | transformer | Transformer | 680 | 329 | 351 | 48.38% | 46.67% | 49.58% | 1.62 pp | -22 | 41 | -0.54 |
| BTC Daily | nn | NN | 680 | 319 | 361 | 46.91% | 43.75% | 49.17% | 3.09 pp | -42 | 41 | -1.02 |
| BTC Daily | lstm | LSTM | 680 | 296 | 384 | 43.53% | 38.75% | 42.50% | 6.47 pp | -88 | 41 | -2.15 |
| BTC Daily | rf | RandomForest | 680 | 292 | 388 | 42.94% | 40.83% | 43.54% | 7.06 pp | -96 | 41 | -2.34 |
| BTC Daily | xgb | XGBoost | 690 | 274 | 416 | 39.71% | 35.00% | 39.58% | 10.29 pp | -142 | 41 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 452 | 219 | 233 | 48.45% | 44.58% | 48.45% | 1.55 pp | -14 | 44 | -0.32 |
| BTC Market Hours | nn | NN | 452 | 213 | 239 | 47.12% | 48.33% | 47.12% | 2.88 pp | -26 | 44 | -0.59 |
| BTC Market Hours | transformer | Transformer | 452 | 209 | 243 | 46.24% | 40.00% | 46.24% | 3.76 pp | -34 | 44 | -0.77 |
| BTC Market Hours | rf | RandomForest | 452 | 196 | 256 | 43.36% | 43.75% | 43.36% | 6.64 pp | -60 | 44 | -1.36 |
| BTC Market Hours | lstm | LSTM | 452 | 191 | 261 | 42.26% | 39.58% | 42.26% | 7.74 pp | -70 | 44 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 452 | 182 | 270 | 40.27% | 38.33% | 40.27% | 9.73 pp | -88 | 44 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 506 | 232 | 274 | 45.85% | 46.25% | 46.25% | 4.15 pp | -42 | 44 | -0.95 |
| BTC Market Hours Daily | nn | NN | 506 | 231 | 275 | 45.65% | 43.33% | 46.46% | 4.35 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 506 | 230 | 276 | 45.45% | 46.25% | 45.62% | 4.55 pp | -46 | 44 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 506 | 210 | 296 | 41.50% | 41.25% | 41.67% | 8.50 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 506 | 202 | 304 | 39.92% | 37.50% | 40.62% | 10.08 pp | -102 | 44 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 506 | 198 | 308 | 39.13% | 35.83% | 38.75% | 10.87 pp | -110 | 44 | -2.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 105 | 53 | 52 | 50.48% | 50.48% | 50.48% | 0.48 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 53 | 52 | 50.48% | 50.48% | 50.48% | 0.48 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 9 | 4 | 69.23% | 69.23% | 69.23% | 19.23 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
