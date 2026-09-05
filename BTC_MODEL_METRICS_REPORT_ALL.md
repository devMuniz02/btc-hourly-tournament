# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T20:28:46.266021+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1249 | 961 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1125 | 760 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 19:00:00+00:00 | 814 | 522 | 291 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 19:00:00+00:00 | 816 | 576 | 238 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 168 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 168 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 168 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 169 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 168 | 83 | 85 | 49.40% | 49.40% | 49.40% | 0.60 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 168 | 83 | 85 | 49.40% | 49.40% | 49.40% | 0.60 pp | -2 | 12 | -0.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 522 | 254 | 268 | 48.66% | 45.83% | 48.75% | 1.34 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 522 | 251 | 271 | 48.08% | 48.33% | 48.54% | 1.92 pp | -20 | 50 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 750 | 365 | 385 | 48.67% | 47.50% | 48.96% | 1.33 pp | -20 | 44 | -0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 12 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 576 | 275 | 301 | 47.74% | 51.25% | 49.17% | 2.26 pp | -26 | 49 | -0.53 |
| BTC Market Hours | nn | NN | 522 | 247 | 275 | 47.32% | 50.83% | 48.54% | 2.68 pp | -28 | 50 | -0.56 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 576 | 269 | 307 | 46.70% | 45.83% | 48.12% | 3.30 pp | -38 | 49 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 927 | 442 | 485 | 47.68% | 49.17% | 46.88% | 2.32 pp | -43 | 49 | -0.88 |
| BTC Daily | transformer | Transformer | 750 | 355 | 395 | 47.33% | 44.58% | 48.75% | 2.67 pp | -40 | 44 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 576 | 265 | 311 | 46.01% | 49.58% | 46.67% | 3.99 pp | -46 | 49 | -0.94 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 927 | 436 | 491 | 47.03% | 46.67% | 45.42% | 2.97 pp | -55 | 49 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 12 | -1.17 |
| BTC Daily | nn | NN | 750 | 348 | 402 | 46.40% | 44.17% | 46.88% | 3.60 pp | -54 | 44 | -1.23 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 522 | 228 | 294 | 43.68% | 43.33% | 44.17% | 6.32 pp | -66 | 50 | -1.32 |
| Consolidated Hourly | nn | NN | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 12 | -1.33 |
| BTC Market Hours | rf | RandomForest | 522 | 225 | 297 | 43.10% | 45.42% | 43.75% | 6.90 pp | -72 | 50 | -1.44 |
| Consolidated Hourly | lstm | LSTM | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 522 | 217 | 305 | 41.57% | 43.75% | 42.29% | 8.43 pp | -88 | 50 | -1.76 |
| BTC Market Hours Daily | rf | RandomForest | 576 | 240 | 336 | 41.67% | 43.33% | 41.25% | 8.33 pp | -96 | 49 | -1.96 |
| Consolidated Hourly | transformer | Transformer | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 12 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 12 | -2.00 |
| BTC Hourly | rf | RandomForest | 927 | 414 | 513 | 44.66% | 44.58% | 44.38% | 5.34 pp | -99 | 49 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 576 | 236 | 340 | 40.97% | 41.25% | 40.83% | 9.03 pp | -104 | 49 | -2.12 |
| BTC Hourly | nn | NN | 927 | 411 | 516 | 44.34% | 42.50% | 42.29% | 5.66 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 576 | 230 | 346 | 39.93% | 41.25% | 39.17% | 10.07 pp | -116 | 49 | -2.37 |
| BTC Daily | lstm | LSTM | 750 | 319 | 431 | 42.53% | 35.83% | 40.83% | 7.47 pp | -112 | 44 | -2.55 |
| BTC Daily | rf | RandomForest | 750 | 316 | 434 | 42.13% | 38.75% | 42.29% | 7.87 pp | -118 | 44 | -2.68 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 927 | 396 | 531 | 42.72% | 37.50% | 41.67% | 7.28 pp | -135 | 49 | -2.76 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 927 | 388 | 539 | 41.86% | 39.58% | 40.62% | 8.14 pp | -151 | 49 | -3.08 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 760 | 298 | 462 | 39.21% | 36.25% | 37.08% | 10.79 pp | -164 | 44 | -3.73 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 927 | 442 | 485 | 47.68% | 49.17% | 46.88% | 2.32 pp | -43 | 49 | -0.88 |
| BTC Hourly | transformer | Transformer | 927 | 436 | 491 | 47.03% | 46.67% | 45.42% | 2.97 pp | -55 | 49 | -1.12 |
| BTC Hourly | rf | RandomForest | 927 | 414 | 513 | 44.66% | 44.58% | 44.38% | 5.34 pp | -99 | 49 | -2.02 |
| BTC Hourly | nn | NN | 927 | 411 | 516 | 44.34% | 42.50% | 42.29% | 5.66 pp | -105 | 49 | -2.14 |
| BTC Hourly | lstm | LSTM | 927 | 396 | 531 | 42.72% | 37.50% | 41.67% | 7.28 pp | -135 | 49 | -2.76 |
| BTC Hourly | xgb | XGBoost | 927 | 388 | 539 | 41.86% | 39.58% | 40.62% | 8.14 pp | -151 | 49 | -3.08 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 750 | 365 | 385 | 48.67% | 47.50% | 48.96% | 1.33 pp | -20 | 44 | -0.45 |
| BTC Daily | transformer | Transformer | 750 | 355 | 395 | 47.33% | 44.58% | 48.75% | 2.67 pp | -40 | 44 | -0.91 |
| BTC Daily | nn | NN | 750 | 348 | 402 | 46.40% | 44.17% | 46.88% | 3.60 pp | -54 | 44 | -1.23 |
| BTC Daily | lstm | LSTM | 750 | 319 | 431 | 42.53% | 35.83% | 40.83% | 7.47 pp | -112 | 44 | -2.55 |
| BTC Daily | rf | RandomForest | 750 | 316 | 434 | 42.13% | 38.75% | 42.29% | 7.87 pp | -118 | 44 | -2.68 |
| BTC Daily | xgb | XGBoost | 760 | 298 | 462 | 39.21% | 36.25% | 37.08% | 10.79 pp | -164 | 44 | -3.73 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 522 | 254 | 268 | 48.66% | 45.83% | 48.75% | 1.34 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 522 | 251 | 271 | 48.08% | 48.33% | 48.54% | 1.92 pp | -20 | 50 | -0.40 |
| BTC Market Hours | nn | NN | 522 | 247 | 275 | 47.32% | 50.83% | 48.54% | 2.68 pp | -28 | 50 | -0.56 |
| BTC Market Hours | lstm | LSTM | 522 | 228 | 294 | 43.68% | 43.33% | 44.17% | 6.32 pp | -66 | 50 | -1.32 |
| BTC Market Hours | rf | RandomForest | 522 | 225 | 297 | 43.10% | 45.42% | 43.75% | 6.90 pp | -72 | 50 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 522 | 217 | 305 | 41.57% | 43.75% | 42.29% | 8.43 pp | -88 | 50 | -1.76 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 576 | 275 | 301 | 47.74% | 51.25% | 49.17% | 2.26 pp | -26 | 49 | -0.53 |
| BTC Market Hours Daily | nn | NN | 576 | 269 | 307 | 46.70% | 45.83% | 48.12% | 3.30 pp | -38 | 49 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 576 | 265 | 311 | 46.01% | 49.58% | 46.67% | 3.99 pp | -46 | 49 | -0.94 |
| BTC Market Hours Daily | rf | RandomForest | 576 | 240 | 336 | 41.67% | 43.33% | 41.25% | 8.33 pp | -96 | 49 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 576 | 236 | 340 | 40.97% | 41.25% | 40.83% | 9.03 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 576 | 230 | 346 | 39.93% | 41.25% | 39.17% | 10.07 pp | -116 | 49 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 168 | 83 | 85 | 49.40% | 49.40% | 49.40% | 0.60 pp | -2 | 12 | -0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 12 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | nn | NN | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 12 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 168 | 83 | 85 | 49.40% | 49.40% | 49.40% | 0.60 pp | -2 | 12 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 168 | 77 | 91 | 45.83% | 45.83% | 45.83% | 4.17 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | nn | NN | 168 | 76 | 92 | 45.24% | 45.24% | 45.24% | 4.76 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 168 | 75 | 93 | 44.64% | 44.64% | 44.64% | 5.36 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 168 | 72 | 96 | 42.86% | 42.86% | 42.86% | 7.14 pp | -24 | 12 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
