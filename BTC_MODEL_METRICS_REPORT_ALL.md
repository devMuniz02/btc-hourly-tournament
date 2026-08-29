# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T14:18:10.694651+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1128 | 840 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1003 | 638 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 13:00:00+00:00 | 595 | 400 | 194 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 13:00:00+00:00 | 597 | 454 | 141 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 58 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 58 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 0 | 58 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 09:00:00+00:00 | 58 | 0 | 58 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 58 | 33 | 25 | 56.90% | 56.90% | 56.90% | 6.90 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 58 | 30 | 28 | 51.72% | 51.72% | 51.72% | 1.72 pp | 2 | 6 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 400 | 198 | 202 | 49.50% | 48.33% | 49.50% | 0.50 pp | -4 | 40 | -0.10 |
| BTC Daily | transformer | Transformer | 628 | 308 | 320 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| Consolidated Hourly | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 58 | 28 | 30 | 48.28% | 48.28% | 48.28% | 1.72 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 628 | 307 | 321 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 400 | 189 | 211 | 47.25% | 50.42% | 47.25% | 2.75 pp | -22 | 40 | -0.55 |
| BTC Market Hours | transformer | Transformer | 400 | 185 | 215 | 46.25% | 42.50% | 46.25% | 3.75 pp | -30 | 40 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 454 | 209 | 245 | 46.04% | 45.42% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 454 | 209 | 245 | 46.04% | 47.50% | 46.04% | 3.96 pp | -36 | 40 | -0.90 |
| BTC Daily | nn | NN | 628 | 295 | 333 | 46.97% | 43.33% | 48.96% | 3.03 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 454 | 207 | 247 | 45.59% | 46.25% | 45.59% | 4.41 pp | -40 | 40 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 806 | 380 | 426 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 806 | 380 | 426 | 47.15% | 44.58% | 46.46% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Market Hours | lstm | LSTM | 400 | 176 | 224 | 44.00% | 45.00% | 44.00% | 6.00 pp | -48 | 40 | -1.20 |
| BTC Market Hours | rf | RandomForest | 400 | 171 | 229 | 42.75% | 41.67% | 42.75% | 7.25 pp | -58 | 40 | -1.45 |
| Consolidated Hourly | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 58 | 24 | 34 | 41.38% | 41.38% | 41.38% | 8.62 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 628 | 279 | 349 | 44.43% | 42.50% | 44.17% | 5.57 pp | -70 | 39 | -1.79 |
| BTC Hourly | nn | NN | 806 | 362 | 444 | 44.91% | 40.00% | 44.58% | 5.09 pp | -82 | 44 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 400 | 161 | 239 | 40.25% | 38.33% | 40.25% | 9.75 pp | -78 | 40 | -1.95 |
| BTC Hourly | rf | RandomForest | 806 | 360 | 446 | 44.67% | 43.75% | 44.38% | 5.33 pp | -86 | 44 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 454 | 185 | 269 | 40.75% | 40.00% | 40.75% | 9.25 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 454 | 184 | 270 | 40.53% | 39.17% | 40.53% | 9.47 pp | -86 | 40 | -2.15 |
| BTC Hourly | lstm | LSTM | 806 | 353 | 453 | 43.80% | 42.50% | 45.00% | 6.20 pp | -100 | 44 | -2.27 |
| BTC Daily | rf | RandomForest | 628 | 268 | 360 | 42.68% | 42.08% | 43.33% | 7.32 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 454 | 178 | 276 | 39.21% | 37.08% | 39.21% | 10.79 pp | -98 | 40 | -2.45 |
| Consolidated Hourly | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 806 | 342 | 464 | 42.43% | 39.58% | 43.33% | 7.57 pp | -122 | 44 | -2.77 |
| BTC Daily | xgb | XGBoost | 638 | 250 | 388 | 39.18% | 31.25% | 39.38% | 10.82 pp | -138 | 39 | -3.54 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 806 | 380 | 426 | 47.15% | 44.17% | 46.88% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | transformer | Transformer | 806 | 380 | 426 | 47.15% | 44.58% | 46.46% | 2.85 pp | -46 | 44 | -1.05 |
| BTC Hourly | nn | NN | 806 | 362 | 444 | 44.91% | 40.00% | 44.58% | 5.09 pp | -82 | 44 | -1.86 |
| BTC Hourly | rf | RandomForest | 806 | 360 | 446 | 44.67% | 43.75% | 44.38% | 5.33 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 806 | 353 | 453 | 43.80% | 42.50% | 45.00% | 6.20 pp | -100 | 44 | -2.27 |
| BTC Hourly | xgb | XGBoost | 806 | 342 | 464 | 42.43% | 39.58% | 43.33% | 7.57 pp | -122 | 44 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 628 | 308 | 320 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 628 | 307 | 321 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 628 | 295 | 333 | 46.97% | 43.33% | 48.96% | 3.03 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 628 | 279 | 349 | 44.43% | 42.50% | 44.17% | 5.57 pp | -70 | 39 | -1.79 |
| BTC Daily | rf | RandomForest | 628 | 268 | 360 | 42.68% | 42.08% | 43.33% | 7.32 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 638 | 250 | 388 | 39.18% | 31.25% | 39.38% | 10.82 pp | -138 | 39 | -3.54 |

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
