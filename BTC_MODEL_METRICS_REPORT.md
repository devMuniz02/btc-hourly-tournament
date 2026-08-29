# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T13:35:28.470095+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1127 | 839 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1003 | 638 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 12:00:00+00:00 | 594 | 400 | 193 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 12:00:00+00:00 | 596 | 454 | 140 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 58 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-23T09:00:00+00:00 | 59 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 400 | 198 | 202 | 49.50% | 48.33% | 49.50% | 0.50 pp | -4 | 40 | -0.10 |
| BTC Daily | transformer | Transformer | 628 | 309 | 319 | 49.20% | 47.92% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 628 | 307 | 321 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 400 | 189 | 211 | 47.25% | 50.42% | 47.25% | 2.75 pp | -22 | 40 | -0.55 |
| BTC Market Hours | transformer | Transformer | 400 | 185 | 215 | 46.25% | 42.50% | 46.25% | 3.75 pp | -30 | 40 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 454 | 209 | 245 | 46.04% | 45.42% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 454 | 209 | 245 | 46.04% | 47.50% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Daily | nn | NN | 628 | 295 | 333 | 46.97% | 43.33% | 48.96% | 3.03 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 454 | 207 | 247 | 45.59% | 46.25% | 45.59% | 4.41 pp | -40 | 40 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 805 | 380 | 425 | 47.20% | 44.58% | 47.08% | 2.80 pp | -45 | 44 | -1.02 |
| BTC Hourly | transformer | Transformer | 805 | 379 | 426 | 47.08% | 44.58% | 46.46% | 2.92 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 400 | 176 | 224 | 44.00% | 45.00% | 44.00% | 6.00 pp | -48 | 40 | -1.20 |
| BTC Market Hours | rf | RandomForest | 400 | 171 | 229 | 42.75% | 41.67% | 42.75% | 7.25 pp | -58 | 40 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 628 | 279 | 349 | 44.43% | 42.50% | 44.17% | 5.57 pp | -70 | 39 | -1.79 |
| BTC Hourly | nn | NN | 805 | 362 | 443 | 44.97% | 40.42% | 44.79% | 5.03 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 805 | 360 | 445 | 44.72% | 44.17% | 44.38% | 5.28 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 400 | 161 | 239 | 40.25% | 38.33% | 40.25% | 9.75 pp | -78 | 40 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 454 | 185 | 269 | 40.75% | 40.00% | 40.75% | 9.25 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 454 | 184 | 270 | 40.53% | 39.17% | 40.53% | 9.47 pp | -86 | 40 | -2.15 |
| BTC Hourly | lstm | LSTM | 805 | 353 | 452 | 43.85% | 42.92% | 45.00% | 6.15 pp | -99 | 44 | -2.25 |
| BTC Daily | rf | RandomForest | 628 | 269 | 359 | 42.83% | 42.50% | 43.54% | 7.17 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 454 | 178 | 276 | 39.21% | 37.08% | 39.21% | 10.79 pp | -98 | 40 | -2.45 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 805 | 341 | 464 | 42.36% | 39.17% | 43.33% | 7.64 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 638 | 251 | 387 | 39.34% | 31.67% | 39.58% | 10.66 pp | -136 | 39 | -3.49 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 805 | 380 | 425 | 47.20% | 44.58% | 47.08% | 2.80 pp | -45 | 44 | -1.02 |
| BTC Hourly | transformer | Transformer | 805 | 379 | 426 | 47.08% | 44.58% | 46.46% | 2.92 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 805 | 362 | 443 | 44.97% | 40.42% | 44.79% | 5.03 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 805 | 360 | 445 | 44.72% | 44.17% | 44.38% | 5.28 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 805 | 353 | 452 | 43.85% | 42.92% | 45.00% | 6.15 pp | -99 | 44 | -2.25 |
| BTC Hourly | xgb | XGBoost | 805 | 341 | 464 | 42.36% | 39.17% | 43.33% | 7.64 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 628 | 309 | 319 | 49.20% | 47.92% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 628 | 307 | 321 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 628 | 295 | 333 | 46.97% | 43.33% | 48.96% | 3.03 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 628 | 279 | 349 | 44.43% | 42.50% | 44.17% | 5.57 pp | -70 | 39 | -1.79 |
| BTC Daily | rf | RandomForest | 628 | 269 | 359 | 42.83% | 42.50% | 43.54% | 7.17 pp | -90 | 39 | -2.31 |
| BTC Daily | xgb | XGBoost | 638 | 251 | 387 | 39.34% | 31.67% | 39.58% | 10.66 pp | -136 | 39 | -3.49 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 400 | 198 | 202 | 49.50% | 48.33% | 49.50% | 0.50 pp | -4 | 40 | -0.10 |
| BTC Market Hours | nn | NN | 400 | 189 | 211 | 47.25% | 50.42% | 47.25% | 2.75 pp | -22 | 40 | -0.55 |
| BTC Market Hours | transformer | Transformer | 400 | 185 | 215 | 46.25% | 42.50% | 46.25% | 3.75 pp | -30 | 40 | -0.75 |
| BTC Market Hours | lstm | LSTM | 400 | 176 | 224 | 44.00% | 45.00% | 44.00% | 6.00 pp | -48 | 40 | -1.20 |
| BTC Market Hours | rf | RandomForest | 400 | 171 | 229 | 42.75% | 41.67% | 42.75% | 7.25 pp | -58 | 40 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 400 | 161 | 239 | 40.25% | 38.33% | 40.25% | 9.75 pp | -78 | 40 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 454 | 209 | 245 | 46.04% | 45.42% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 454 | 209 | 245 | 46.04% | 47.50% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | nn | NN | 454 | 207 | 247 | 45.59% | 46.25% | 45.59% | 4.41 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 454 | 185 | 269 | 40.75% | 40.00% | 40.75% | 9.25 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 454 | 184 | 270 | 40.53% | 39.17% | 40.53% | 9.47 pp | -86 | 40 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 454 | 178 | 276 | 39.21% | 37.08% | 39.21% | 10.79 pp | -98 | 40 | -2.45 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
