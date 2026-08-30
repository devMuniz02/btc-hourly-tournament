# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T13:20:05.965274+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1145 | 857 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1021 | 656 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 625 | 418 | 206 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 627 | 472 | 153 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 418 | 207 | 211 | 49.52% | 47.50% | 49.52% | 0.48 pp | -4 | 42 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 646 | 314 | 332 | 48.61% | 46.25% | 50.00% | 1.39 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 646 | 313 | 333 | 48.45% | 45.42% | 49.38% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 418 | 197 | 221 | 47.13% | 50.42% | 47.13% | 2.87 pp | -24 | 42 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 472 | 219 | 253 | 46.40% | 47.08% | 46.40% | 3.60 pp | -34 | 41 | -0.83 |
| BTC Market Hours | transformer | Transformer | 418 | 191 | 227 | 45.69% | 41.25% | 45.69% | 4.31 pp | -36 | 42 | -0.86 |
| BTC Hourly | transformer | Transformer | 823 | 391 | 432 | 47.51% | 47.08% | 46.67% | 2.49 pp | -41 | 44 | -0.93 |
| BTC Daily | nn | NN | 646 | 303 | 343 | 46.90% | 42.50% | 49.38% | 3.10 pp | -40 | 40 | -1.00 |
| BTC Market Hours Daily | nn | NN | 472 | 215 | 257 | 45.55% | 44.58% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 472 | 215 | 257 | 45.55% | 45.00% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 823 | 386 | 437 | 46.90% | 42.50% | 46.67% | 3.10 pp | -51 | 44 | -1.16 |
| BTC Market Hours | lstm | LSTM | 418 | 183 | 235 | 43.78% | 43.75% | 43.78% | 6.22 pp | -52 | 42 | -1.24 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| BTC Market Hours | rf | RandomForest | 418 | 181 | 237 | 43.30% | 42.92% | 43.30% | 6.70 pp | -56 | 42 | -1.33 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 823 | 372 | 451 | 45.20% | 42.50% | 44.79% | 4.80 pp | -79 | 44 | -1.80 |
| BTC Daily | lstm | LSTM | 646 | 286 | 360 | 44.27% | 42.08% | 43.75% | 5.73 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 472 | 196 | 276 | 41.53% | 42.50% | 41.53% | 8.47 pp | -80 | 41 | -1.95 |
| BTC Hourly | rf | RandomForest | 823 | 368 | 455 | 44.71% | 44.58% | 44.58% | 5.29 pp | -87 | 44 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 418 | 167 | 251 | 39.95% | 37.92% | 39.95% | 10.05 pp | -84 | 42 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 472 | 192 | 280 | 40.68% | 38.75% | 40.68% | 9.32 pp | -88 | 41 | -2.15 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| BTC Hourly | lstm | LSTM | 823 | 359 | 464 | 43.62% | 42.08% | 44.17% | 6.38 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 646 | 275 | 371 | 42.57% | 41.25% | 43.33% | 7.43 pp | -96 | 40 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 472 | 184 | 288 | 38.98% | 35.83% | 38.98% | 11.02 pp | -104 | 41 | -2.54 |
| BTC Hourly | xgb | XGBoost | 823 | 347 | 476 | 42.16% | 39.17% | 42.50% | 7.84 pp | -129 | 44 | -2.93 |
| BTC Daily | xgb | XGBoost | 656 | 258 | 398 | 39.33% | 31.67% | 39.38% | 10.67 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 823 | 391 | 432 | 47.51% | 47.08% | 46.67% | 2.49 pp | -41 | 44 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 823 | 386 | 437 | 46.90% | 42.50% | 46.67% | 3.10 pp | -51 | 44 | -1.16 |
| BTC Hourly | nn | NN | 823 | 372 | 451 | 45.20% | 42.50% | 44.79% | 4.80 pp | -79 | 44 | -1.80 |
| BTC Hourly | rf | RandomForest | 823 | 368 | 455 | 44.71% | 44.58% | 44.58% | 5.29 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 823 | 359 | 464 | 43.62% | 42.08% | 44.17% | 6.38 pp | -105 | 44 | -2.39 |
| BTC Hourly | xgb | XGBoost | 823 | 347 | 476 | 42.16% | 39.17% | 42.50% | 7.84 pp | -129 | 44 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 646 | 314 | 332 | 48.61% | 46.25% | 50.00% | 1.39 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 646 | 313 | 333 | 48.45% | 45.42% | 49.38% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 646 | 303 | 343 | 46.90% | 42.50% | 49.38% | 3.10 pp | -40 | 40 | -1.00 |
| BTC Daily | lstm | LSTM | 646 | 286 | 360 | 44.27% | 42.08% | 43.75% | 5.73 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 646 | 275 | 371 | 42.57% | 41.25% | 43.33% | 7.43 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 656 | 258 | 398 | 39.33% | 31.67% | 39.38% | 10.67 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 418 | 207 | 211 | 49.52% | 47.50% | 49.52% | 0.48 pp | -4 | 42 | -0.10 |
| BTC Market Hours | nn | NN | 418 | 197 | 221 | 47.13% | 50.42% | 47.13% | 2.87 pp | -24 | 42 | -0.57 |
| BTC Market Hours | transformer | Transformer | 418 | 191 | 227 | 45.69% | 41.25% | 45.69% | 4.31 pp | -36 | 42 | -0.86 |
| BTC Market Hours | lstm | LSTM | 418 | 183 | 235 | 43.78% | 43.75% | 43.78% | 6.22 pp | -52 | 42 | -1.24 |
| BTC Market Hours | rf | RandomForest | 418 | 181 | 237 | 43.30% | 42.92% | 43.30% | 6.70 pp | -56 | 42 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 418 | 167 | 251 | 39.95% | 37.92% | 39.95% | 10.05 pp | -84 | 42 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 472 | 219 | 253 | 46.40% | 47.08% | 46.40% | 3.60 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | nn | NN | 472 | 215 | 257 | 45.55% | 44.58% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 472 | 215 | 257 | 45.55% | 45.00% | 45.55% | 4.45 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 472 | 196 | 276 | 41.53% | 42.50% | 41.53% | 8.47 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 472 | 192 | 280 | 40.68% | 38.75% | 40.68% | 9.32 pp | -88 | 41 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 472 | 184 | 288 | 38.98% | 35.83% | 38.98% | 11.02 pp | -104 | 41 | -2.54 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

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
