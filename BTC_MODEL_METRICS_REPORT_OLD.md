# BTC Model Metrics Report - Old Baseline

Generated at: 2026-08-21T14:16:59.900819+00:00
Scope: `old`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-04-27 23:00:00+00:00 | 1017 | 789 | 228 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-04-27 22:00:00+00:00 | 857 | 552 | 304 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-04-27 23:00:00+00:00 | 441 | 326 | 114 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-04-27 22:00:00+00:00 | 443 | 380 | 61 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 06:00:00+00:00 to 2026-05-18 06:00:00+00:00 | 1 | 1 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 06:00:00+00:00 to 2026-05-18 06:00:00+00:00 | 1 | 1 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 06:00:00+00:00 to 2026-05-18 06:00:00+00:00 | 1 | 1 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 06:00:00+00:00 to 2026-05-18 06:00:00+00:00 | 1 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 287 | 265 | 51.99% | 52.92% | 52.29% | 1.99 pp | 22 | 35 | 0.63 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 167 | 159 | 51.23% | 48.75% | 51.23% | 1.23 pp | 8 | 34 | 0.24 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 376 | 379 | 49.80% | 46.67% | 51.04% | 0.20 pp | -3 | 41 | -0.07 |
| BTC Market Hours Daily | nn | NN | 380 | 180 | 200 | 47.37% | 47.92% | 47.37% | 2.63 pp | -20 | 34 | -0.59 |
| BTC Daily | nn | NN | 552 | 264 | 288 | 47.83% | 45.00% | 47.29% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Market Hours | transformer | Transformer | 326 | 151 | 175 | 46.32% | 46.67% | 46.32% | 3.68 pp | -24 | 34 | -0.71 |
| BTC Market Hours | nn | NN | 326 | 149 | 177 | 45.71% | 47.92% | 45.71% | 4.29 pp | -28 | 34 | -0.82 |
| BTC Daily | lstm | LSTM | 552 | 261 | 291 | 47.28% | 47.50% | 46.88% | 2.72 pp | -30 | 35 | -0.86 |
| BTC Daily | transformer | Transformer | 552 | 260 | 292 | 47.10% | 48.75% | 47.50% | 2.90 pp | -32 | 35 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 174 | 206 | 45.79% | 46.25% | 45.79% | 4.21 pp | -32 | 34 | -0.94 |
| BTC Hourly | transformer | Transformer | 755 | 356 | 399 | 47.15% | 42.08% | 45.62% | 2.85 pp | -43 | 41 | -1.05 |
| BTC Hourly | nn | NN | 755 | 355 | 400 | 47.02% | 42.08% | 46.88% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Market Hours | lstm | LSTM | 326 | 143 | 183 | 43.87% | 45.83% | 43.87% | 6.13 pp | -40 | 34 | -1.18 |
| BTC Hourly | rf | RandomForest | 755 | 351 | 404 | 46.49% | 44.58% | 46.25% | 3.51 pp | -53 | 41 | -1.29 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 167 | 213 | 43.95% | 42.92% | 43.95% | 6.05 pp | -46 | 34 | -1.35 |
| BTC Daily | rf | RandomForest | 552 | 248 | 304 | 44.93% | 45.00% | 44.58% | 5.07 pp | -56 | 35 | -1.60 |
| BTC Hourly | lstm | LSTM | 755 | 344 | 411 | 45.56% | 43.33% | 47.50% | 4.44 pp | -67 | 41 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 326 | 134 | 192 | 41.10% | 41.67% | 41.10% | 8.90 pp | -58 | 34 | -1.71 |
| BTC Market Hours | rf | RandomForest | 326 | 132 | 194 | 40.49% | 40.42% | 40.49% | 9.51 pp | -62 | 34 | -1.82 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 158 | 222 | 41.58% | 40.83% | 41.58% | 8.42 pp | -64 | 34 | -1.88 |
| BTC Hourly | xgb | XGBoost | 755 | 337 | 418 | 44.64% | 44.17% | 46.46% | 5.36 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 156 | 224 | 41.05% | 39.58% | 41.05% | 8.95 pp | -68 | 34 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 156 | 224 | 41.05% | 41.25% | 41.05% | 8.95 pp | -68 | 34 | -2.00 |
| BTC Daily | xgb | XGBoost | 552 | 229 | 323 | 41.49% | 38.75% | 41.46% | 8.51 pp | -94 | 35 | -2.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 376 | 379 | 49.80% | 46.67% | 51.04% | 0.20 pp | -3 | 41 | -0.07 |
| BTC Hourly | transformer | Transformer | 755 | 356 | 399 | 47.15% | 42.08% | 45.62% | 2.85 pp | -43 | 41 | -1.05 |
| BTC Hourly | nn | NN | 755 | 355 | 400 | 47.02% | 42.08% | 46.88% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Hourly | rf | RandomForest | 755 | 351 | 404 | 46.49% | 44.58% | 46.25% | 3.51 pp | -53 | 41 | -1.29 |
| BTC Hourly | lstm | LSTM | 755 | 344 | 411 | 45.56% | 43.33% | 47.50% | 4.44 pp | -67 | 41 | -1.63 |
| BTC Hourly | xgb | XGBoost | 755 | 337 | 418 | 44.64% | 44.17% | 46.46% | 5.36 pp | -81 | 41 | -1.98 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 287 | 265 | 51.99% | 52.92% | 52.29% | 1.99 pp | 22 | 35 | 0.63 |
| BTC Daily | nn | NN | 552 | 264 | 288 | 47.83% | 45.00% | 47.29% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Daily | lstm | LSTM | 552 | 261 | 291 | 47.28% | 47.50% | 46.88% | 2.72 pp | -30 | 35 | -0.86 |
| BTC Daily | transformer | Transformer | 552 | 260 | 292 | 47.10% | 48.75% | 47.50% | 2.90 pp | -32 | 35 | -0.91 |
| BTC Daily | rf | RandomForest | 552 | 248 | 304 | 44.93% | 45.00% | 44.58% | 5.07 pp | -56 | 35 | -1.60 |
| BTC Daily | xgb | XGBoost | 552 | 229 | 323 | 41.49% | 38.75% | 41.46% | 8.51 pp | -94 | 35 | -2.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 167 | 159 | 51.23% | 48.75% | 51.23% | 1.23 pp | 8 | 34 | 0.24 |
| BTC Market Hours | transformer | Transformer | 326 | 151 | 175 | 46.32% | 46.67% | 46.32% | 3.68 pp | -24 | 34 | -0.71 |
| BTC Market Hours | nn | NN | 326 | 149 | 177 | 45.71% | 47.92% | 45.71% | 4.29 pp | -28 | 34 | -0.82 |
| BTC Market Hours | lstm | LSTM | 326 | 143 | 183 | 43.87% | 45.83% | 43.87% | 6.13 pp | -40 | 34 | -1.18 |
| BTC Market Hours | xgb | XGBoost | 326 | 134 | 192 | 41.10% | 41.67% | 41.10% | 8.90 pp | -58 | 34 | -1.71 |
| BTC Market Hours | rf | RandomForest | 326 | 132 | 194 | 40.49% | 40.42% | 40.49% | 9.51 pp | -62 | 34 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 380 | 180 | 200 | 47.37% | 47.92% | 47.37% | 2.63 pp | -20 | 34 | -0.59 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 174 | 206 | 45.79% | 46.25% | 45.79% | 4.21 pp | -32 | 34 | -0.94 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 167 | 213 | 43.95% | 42.92% | 43.95% | 6.05 pp | -46 | 34 | -1.35 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 158 | 222 | 41.58% | 40.83% | 41.58% | 8.42 pp | -64 | 34 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 156 | 224 | 41.05% | 39.58% | 41.05% | 8.95 pp | -68 | 34 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 156 | 224 | 41.05% | 41.25% | 41.05% | 8.95 pp | -68 | 34 | -2.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

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
