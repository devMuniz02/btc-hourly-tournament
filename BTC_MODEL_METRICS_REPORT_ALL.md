# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T16:36:30.363273+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 792 | 327 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 955 | 590 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 505 | 352 | 152 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 507 | 406 | 99 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 16 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 16 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 0 | 16 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 0 | 16 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 352 | 173 | 179 | 49.15% | 47.08% | 49.15% | 0.85 pp | -6 | 36 | -0.17 |
| BTC Daily | transformer | Transformer | 580 | 286 | 294 | 49.31% | 52.08% | 49.38% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 580 | 285 | 295 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| BTC Market Hours | transformer | Transformer | 352 | 166 | 186 | 47.16% | 45.83% | 47.16% | 2.84 pp | -20 | 36 | -0.56 |
| BTC Market Hours Daily | nn | NN | 406 | 187 | 219 | 46.06% | 47.92% | 46.06% | 3.94 pp | -32 | 36 | -0.89 |
| BTC Market Hours | nn | NN | 352 | 160 | 192 | 45.45% | 47.50% | 45.45% | 4.55 pp | -32 | 36 | -0.89 |
| BTC Daily | nn | NN | 580 | 273 | 307 | 47.07% | 45.42% | 47.92% | 2.93 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 406 | 186 | 220 | 45.81% | 45.83% | 45.81% | 4.19 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 406 | 185 | 221 | 45.57% | 47.08% | 45.57% | 4.43 pp | -36 | 36 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 758 | 355 | 403 | 46.83% | 42.92% | 47.29% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 758 | 355 | 403 | 46.83% | 44.17% | 45.62% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | rf | RandomForest | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Daily | lstm | LSTM | 580 | 261 | 319 | 45.00% | 45.83% | 44.79% | 5.00 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 352 | 146 | 206 | 41.48% | 42.08% | 41.48% | 8.52 pp | -60 | 36 | -1.67 |
| BTC Hourly | rf | RandomForest | 758 | 340 | 418 | 44.85% | 45.00% | 44.58% | 5.15 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 758 | 338 | 420 | 44.59% | 40.83% | 45.00% | 5.41 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 406 | 167 | 239 | 41.13% | 39.58% | 41.13% | 8.87 pp | -72 | 36 | -2.00 |
| BTC Daily | rf | RandomForest | 580 | 250 | 330 | 43.10% | 44.17% | 43.75% | 6.90 pp | -80 | 37 | -2.16 |
| BTC Hourly | lstm | LSTM | 758 | 333 | 425 | 43.93% | 42.50% | 45.42% | 6.07 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 37.92% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Hourly | xgb | XGBoost | 758 | 326 | 432 | 43.01% | 42.08% | 44.38% | 6.99 pp | -106 | 42 | -2.52 |
| BTC Daily | xgb | XGBoost | 590 | 236 | 354 | 40.00% | 34.58% | 40.42% | 10.00 pp | -118 | 37 | -3.19 |
| Consolidated Hourly | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 758 | 355 | 403 | 46.83% | 42.92% | 47.29% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | transformer | Transformer | 758 | 355 | 403 | 46.83% | 44.17% | 45.62% | 3.17 pp | -48 | 42 | -1.14 |
| BTC Hourly | rf | RandomForest | 758 | 340 | 418 | 44.85% | 45.00% | 44.58% | 5.15 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 758 | 338 | 420 | 44.59% | 40.83% | 45.00% | 5.41 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 758 | 333 | 425 | 43.93% | 42.50% | 45.42% | 6.07 pp | -92 | 42 | -2.19 |
| BTC Hourly | xgb | XGBoost | 758 | 326 | 432 | 43.01% | 42.08% | 44.38% | 6.99 pp | -106 | 42 | -2.52 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 580 | 286 | 294 | 49.31% | 52.08% | 49.38% | 0.69 pp | -8 | 37 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 580 | 285 | 295 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 580 | 273 | 307 | 47.07% | 45.42% | 47.92% | 2.93 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 580 | 261 | 319 | 45.00% | 45.83% | 44.79% | 5.00 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 580 | 250 | 330 | 43.10% | 44.17% | 43.75% | 6.90 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 590 | 236 | 354 | 40.00% | 34.58% | 40.42% | 10.00 pp | -118 | 37 | -3.19 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 352 | 173 | 179 | 49.15% | 47.08% | 49.15% | 0.85 pp | -6 | 36 | -0.17 |
| BTC Market Hours | transformer | Transformer | 352 | 166 | 186 | 47.16% | 45.83% | 47.16% | 2.84 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 352 | 160 | 192 | 45.45% | 47.50% | 45.45% | 4.55 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | rf | RandomForest | 352 | 150 | 202 | 42.61% | 42.50% | 42.61% | 7.39 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 352 | 146 | 206 | 41.48% | 42.08% | 41.48% | 8.52 pp | -60 | 36 | -1.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 406 | 187 | 219 | 46.06% | 47.92% | 46.06% | 3.94 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 406 | 186 | 220 | 45.81% | 45.83% | 45.81% | 4.19 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 406 | 185 | 221 | 45.57% | 47.08% | 45.57% | 4.43 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 406 | 167 | 239 | 41.13% | 39.58% | 41.13% | 8.87 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 37.92% | 40.15% | 9.85 pp | -80 | 36 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

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
