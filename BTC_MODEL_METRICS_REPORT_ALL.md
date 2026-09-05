# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T03:34:09.017734+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1237 | 949 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1113 | 748 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 794 | 510 | 283 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 796 | 564 | 230 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 157 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T12:00:00+00:00 | 158 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 510 | 247 | 263 | 48.43% | 45.42% | 48.54% | 1.57 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 510 | 245 | 265 | 48.04% | 46.67% | 48.33% | 1.96 pp | -20 | 49 | -0.41 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 738 | 357 | 381 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 44 | -0.55 |
| BTC Market Hours Daily | transformer | Transformer | 564 | 268 | 296 | 47.52% | 51.25% | 48.75% | 2.48 pp | -28 | 49 | -0.57 |
| BTC Market Hours | nn | NN | 510 | 240 | 270 | 47.06% | 49.58% | 47.92% | 2.94 pp | -30 | 49 | -0.61 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 738 | 352 | 386 | 47.70% | 46.25% | 49.58% | 2.30 pp | -34 | 44 | -0.77 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 915 | 438 | 477 | 47.87% | 50.00% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Market Hours Daily | nn | NN | 564 | 261 | 303 | 46.28% | 45.83% | 47.71% | 3.72 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 564 | 260 | 304 | 46.10% | 48.75% | 46.46% | 3.90 pp | -44 | 49 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 915 | 432 | 483 | 47.21% | 47.50% | 46.25% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Daily | nn | NN | 738 | 342 | 396 | 46.34% | 44.17% | 46.88% | 3.66 pp | -54 | 44 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 510 | 222 | 288 | 43.53% | 43.33% | 43.54% | 6.47 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 510 | 220 | 290 | 43.14% | 44.17% | 43.33% | 6.86 pp | -70 | 49 | -1.43 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 510 | 210 | 300 | 41.18% | 42.92% | 41.67% | 8.82 pp | -90 | 49 | -1.84 |
| Consolidated Hourly | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | rf | RandomForest | 564 | 234 | 330 | 41.49% | 42.08% | 40.42% | 8.51 pp | -96 | 49 | -1.96 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 17 | 25 | 40.48% | 40.48% | 40.48% | 9.52 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 915 | 407 | 508 | 44.48% | 43.75% | 44.17% | 5.52 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 915 | 406 | 509 | 44.37% | 42.92% | 42.08% | 5.63 pp | -103 | 48 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 564 | 229 | 335 | 40.60% | 39.58% | 40.83% | 9.40 pp | -106 | 49 | -2.16 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 564 | 226 | 338 | 40.07% | 40.83% | 39.17% | 9.93 pp | -112 | 49 | -2.29 |
| BTC Daily | lstm | LSTM | 738 | 317 | 421 | 42.95% | 36.67% | 41.04% | 7.05 pp | -104 | 44 | -2.36 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | rf | RandomForest | 738 | 313 | 425 | 42.41% | 40.00% | 42.92% | 7.59 pp | -112 | 44 | -2.55 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 915 | 391 | 524 | 42.73% | 39.17% | 41.46% | 7.27 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 915 | 382 | 533 | 41.75% | 39.58% | 40.00% | 8.25 pp | -151 | 48 | -3.15 |
| BTC Daily | xgb | XGBoost | 748 | 296 | 452 | 39.57% | 36.25% | 37.71% | 10.43 pp | -156 | 44 | -3.55 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 915 | 438 | 477 | 47.87% | 50.00% | 47.71% | 2.13 pp | -39 | 48 | -0.81 |
| BTC Hourly | transformer | Transformer | 915 | 432 | 483 | 47.21% | 47.50% | 46.25% | 2.79 pp | -51 | 48 | -1.06 |
| BTC Hourly | rf | RandomForest | 915 | 407 | 508 | 44.48% | 43.75% | 44.17% | 5.52 pp | -101 | 48 | -2.10 |
| BTC Hourly | nn | NN | 915 | 406 | 509 | 44.37% | 42.92% | 42.08% | 5.63 pp | -103 | 48 | -2.15 |
| BTC Hourly | lstm | LSTM | 915 | 391 | 524 | 42.73% | 39.17% | 41.46% | 7.27 pp | -133 | 48 | -2.77 |
| BTC Hourly | xgb | XGBoost | 915 | 382 | 533 | 41.75% | 39.58% | 40.00% | 8.25 pp | -151 | 48 | -3.15 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 738 | 357 | 381 | 48.37% | 47.08% | 48.33% | 1.63 pp | -24 | 44 | -0.55 |
| BTC Daily | transformer | Transformer | 738 | 352 | 386 | 47.70% | 46.25% | 49.58% | 2.30 pp | -34 | 44 | -0.77 |
| BTC Daily | nn | NN | 738 | 342 | 396 | 46.34% | 44.17% | 46.88% | 3.66 pp | -54 | 44 | -1.23 |
| BTC Daily | lstm | LSTM | 738 | 317 | 421 | 42.95% | 36.67% | 41.04% | 7.05 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 738 | 313 | 425 | 42.41% | 40.00% | 42.92% | 7.59 pp | -112 | 44 | -2.55 |
| BTC Daily | xgb | XGBoost | 748 | 296 | 452 | 39.57% | 36.25% | 37.71% | 10.43 pp | -156 | 44 | -3.55 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 510 | 247 | 263 | 48.43% | 45.42% | 48.54% | 1.57 pp | -16 | 49 | -0.33 |
| BTC Market Hours | transformer | Transformer | 510 | 245 | 265 | 48.04% | 46.67% | 48.33% | 1.96 pp | -20 | 49 | -0.41 |
| BTC Market Hours | nn | NN | 510 | 240 | 270 | 47.06% | 49.58% | 47.92% | 2.94 pp | -30 | 49 | -0.61 |
| BTC Market Hours | lstm | LSTM | 510 | 222 | 288 | 43.53% | 43.33% | 43.54% | 6.47 pp | -66 | 49 | -1.35 |
| BTC Market Hours | rf | RandomForest | 510 | 220 | 290 | 43.14% | 44.17% | 43.33% | 6.86 pp | -70 | 49 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 510 | 210 | 300 | 41.18% | 42.92% | 41.67% | 8.82 pp | -90 | 49 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 564 | 268 | 296 | 47.52% | 51.25% | 48.75% | 2.48 pp | -28 | 49 | -0.57 |
| BTC Market Hours Daily | nn | NN | 564 | 261 | 303 | 46.28% | 45.83% | 47.71% | 3.72 pp | -42 | 49 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 564 | 260 | 304 | 46.10% | 48.75% | 46.46% | 3.90 pp | -44 | 49 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 564 | 234 | 330 | 41.49% | 42.08% | 40.42% | 8.51 pp | -96 | 49 | -1.96 |
| BTC Market Hours Daily | lstm | LSTM | 564 | 229 | 335 | 40.60% | 39.58% | 40.83% | 9.40 pp | -106 | 49 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 564 | 226 | 338 | 40.07% | 40.83% | 39.17% | 9.93 pp | -112 | 49 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 157 | 73 | 84 | 46.50% | 46.50% | 46.50% | 3.50 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 157 | 71 | 86 | 45.22% | 45.22% | 45.22% | 4.78 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 157 | 67 | 90 | 42.68% | 42.68% | 42.68% | 7.32 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours | rf | RandomForest | 41 | 19 | 22 | 46.34% | 46.34% | 46.34% | 3.66 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 42 | 17 | 25 | 40.48% | 40.48% | 40.48% | 9.52 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
