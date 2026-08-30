# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T15:02:54.050164+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1147 | 859 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1023 | 658 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 629 | 420 | 208 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 14:00:00+00:00 | 631 | 474 | 155 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T03:00:00+00:00 | 74 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T03:00:00+00:00 | 74 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T03:00:00+00:00 | 74 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T03:00:00+00:00 | 75 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 420 | 207 | 213 | 49.29% | 47.08% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 648 | 316 | 332 | 48.77% | 46.67% | 50.00% | 1.23 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 648 | 314 | 334 | 48.46% | 45.42% | 49.38% | 1.54 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 420 | 197 | 223 | 46.90% | 50.00% | 46.90% | 3.10 pp | -26 | 42 | -0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 474 | 219 | 255 | 46.20% | 46.67% | 46.20% | 3.80 pp | -36 | 42 | -0.86 |
| BTC Market Hours | transformer | Transformer | 420 | 191 | 229 | 45.48% | 40.83% | 45.48% | 4.52 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 825 | 392 | 433 | 47.52% | 47.50% | 46.67% | 2.48 pp | -41 | 44 | -0.93 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 474 | 215 | 259 | 45.36% | 44.58% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 474 | 215 | 259 | 45.36% | 45.00% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 648 | 303 | 345 | 46.76% | 42.08% | 48.96% | 3.24 pp | -42 | 40 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 825 | 386 | 439 | 46.79% | 42.50% | 46.46% | 3.21 pp | -53 | 44 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 420 | 183 | 237 | 43.57% | 42.92% | 43.57% | 6.43 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 420 | 181 | 239 | 43.10% | 42.50% | 43.10% | 6.90 pp | -58 | 42 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 825 | 372 | 453 | 45.09% | 42.50% | 44.79% | 4.91 pp | -81 | 44 | -1.84 |
| BTC Daily | lstm | LSTM | 648 | 286 | 362 | 44.14% | 41.67% | 43.54% | 5.86 pp | -76 | 40 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 474 | 196 | 278 | 41.35% | 42.50% | 41.35% | 8.65 pp | -82 | 42 | -1.95 |
| BTC Hourly | rf | RandomForest | 825 | 368 | 457 | 44.61% | 43.75% | 44.17% | 5.39 pp | -89 | 44 | -2.02 |
| BTC Market Hours | xgb | XGBoost | 420 | 167 | 253 | 39.76% | 37.50% | 39.76% | 10.24 pp | -86 | 42 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 474 | 192 | 282 | 40.51% | 38.75% | 40.51% | 9.49 pp | -90 | 42 | -2.14 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 648 | 277 | 371 | 42.75% | 41.67% | 43.33% | 7.25 pp | -94 | 40 | -2.35 |
| BTC Hourly | lstm | LSTM | 825 | 359 | 466 | 43.52% | 41.67% | 43.75% | 6.48 pp | -107 | 44 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 474 | 184 | 290 | 38.82% | 35.42% | 38.82% | 11.18 pp | -106 | 42 | -2.52 |
| BTC Hourly | xgb | XGBoost | 825 | 348 | 477 | 42.18% | 39.17% | 42.50% | 7.82 pp | -129 | 44 | -2.93 |
| BTC Daily | xgb | XGBoost | 658 | 260 | 398 | 39.51% | 32.50% | 39.79% | 10.49 pp | -138 | 40 | -3.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 825 | 392 | 433 | 47.52% | 47.50% | 46.67% | 2.48 pp | -41 | 44 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 825 | 386 | 439 | 46.79% | 42.50% | 46.46% | 3.21 pp | -53 | 44 | -1.20 |
| BTC Hourly | nn | NN | 825 | 372 | 453 | 45.09% | 42.50% | 44.79% | 4.91 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 825 | 368 | 457 | 44.61% | 43.75% | 44.17% | 5.39 pp | -89 | 44 | -2.02 |
| BTC Hourly | lstm | LSTM | 825 | 359 | 466 | 43.52% | 41.67% | 43.75% | 6.48 pp | -107 | 44 | -2.43 |
| BTC Hourly | xgb | XGBoost | 825 | 348 | 477 | 42.18% | 39.17% | 42.50% | 7.82 pp | -129 | 44 | -2.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 648 | 316 | 332 | 48.77% | 46.67% | 50.00% | 1.23 pp | -16 | 40 | -0.40 |
| BTC Daily | transformer | Transformer | 648 | 314 | 334 | 48.46% | 45.42% | 49.38% | 1.54 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 648 | 303 | 345 | 46.76% | 42.08% | 48.96% | 3.24 pp | -42 | 40 | -1.05 |
| BTC Daily | lstm | LSTM | 648 | 286 | 362 | 44.14% | 41.67% | 43.54% | 5.86 pp | -76 | 40 | -1.90 |
| BTC Daily | rf | RandomForest | 648 | 277 | 371 | 42.75% | 41.67% | 43.33% | 7.25 pp | -94 | 40 | -2.35 |
| BTC Daily | xgb | XGBoost | 658 | 260 | 398 | 39.51% | 32.50% | 39.79% | 10.49 pp | -138 | 40 | -3.45 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 420 | 207 | 213 | 49.29% | 47.08% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Market Hours | nn | NN | 420 | 197 | 223 | 46.90% | 50.00% | 46.90% | 3.10 pp | -26 | 42 | -0.62 |
| BTC Market Hours | transformer | Transformer | 420 | 191 | 229 | 45.48% | 40.83% | 45.48% | 4.52 pp | -38 | 42 | -0.90 |
| BTC Market Hours | lstm | LSTM | 420 | 183 | 237 | 43.57% | 42.92% | 43.57% | 6.43 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 420 | 181 | 239 | 43.10% | 42.50% | 43.10% | 6.90 pp | -58 | 42 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 420 | 167 | 253 | 39.76% | 37.50% | 39.76% | 10.24 pp | -86 | 42 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 474 | 219 | 255 | 46.20% | 46.67% | 46.20% | 3.80 pp | -36 | 42 | -0.86 |
| BTC Market Hours Daily | nn | NN | 474 | 215 | 259 | 45.36% | 44.58% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 474 | 215 | 259 | 45.36% | 45.00% | 45.36% | 4.64 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 474 | 196 | 278 | 41.35% | 42.50% | 41.35% | 8.65 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 474 | 192 | 282 | 40.51% | 38.75% | 40.51% | 9.49 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 474 | 184 | 290 | 38.82% | 35.42% | 38.82% | 11.18 pp | -106 | 42 | -2.52 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 74 | 40 | 34 | 54.05% | 54.05% | 54.05% | 4.05 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 74 | 32 | 42 | 43.24% | 43.24% | 43.24% | 6.76 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 74 | 31 | 43 | 41.89% | 41.89% | 41.89% | 8.11 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 74 | 28 | 46 | 37.84% | 37.84% | 37.84% | 12.16 pp | -18 | 8 | -2.25 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
