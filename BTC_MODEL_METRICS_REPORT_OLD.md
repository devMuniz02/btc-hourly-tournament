# BTC Model Metrics Report - Old Baseline

Generated at: 2026-08-31T06:23:08.837934+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 287 | 265 | 51.99% | 52.50% | 52.50% | 1.99 pp | 22 | 35 | 0.63 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 171 | 155 | 52.45% | 51.25% | 52.45% | 2.45 pp | 16 | 34 | 0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 373 | 382 | 49.40% | 46.25% | 50.62% | 0.60 pp | -9 | 41 | -0.22 |
| BTC Market Hours | nn | NN | 326 | 155 | 171 | 47.55% | 48.75% | 47.55% | 2.45 pp | -16 | 34 | -0.47 |
| BTC Market Hours | transformer | Transformer | 326 | 153 | 173 | 46.93% | 47.50% | 46.93% | 3.07 pp | -20 | 34 | -0.59 |
| BTC Market Hours Daily | nn | NN | 380 | 179 | 201 | 47.11% | 46.25% | 47.11% | 2.89 pp | -22 | 34 | -0.65 |
| BTC Daily | nn | NN | 552 | 264 | 288 | 47.83% | 45.42% | 47.50% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Daily | transformer | Transformer | 552 | 264 | 288 | 47.83% | 50.00% | 48.54% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 176 | 204 | 46.32% | 44.58% | 46.32% | 3.68 pp | -28 | 34 | -0.82 |
| BTC Market Hours | lstm | LSTM | 326 | 149 | 177 | 45.71% | 45.83% | 45.71% | 4.29 pp | -28 | 34 | -0.82 |
| BTC Daily | lstm | LSTM | 552 | 261 | 291 | 47.28% | 47.08% | 46.67% | 2.72 pp | -30 | 35 | -0.86 |
| BTC Hourly | transformer | Transformer | 755 | 357 | 398 | 47.28% | 42.50% | 46.04% | 2.72 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 173 | 207 | 45.53% | 47.08% | 45.53% | 4.47 pp | -34 | 34 | -1.00 |
| BTC Hourly | rf | RandomForest | 755 | 356 | 399 | 47.15% | 45.83% | 47.08% | 2.85 pp | -43 | 41 | -1.05 |
| BTC Hourly | nn | NN | 755 | 354 | 401 | 46.89% | 42.50% | 46.88% | 3.11 pp | -47 | 41 | -1.15 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 165 | 215 | 43.42% | 41.25% | 43.42% | 6.58 pp | -50 | 34 | -1.47 |
| BTC Market Hours | rf | RandomForest | 326 | 138 | 188 | 42.33% | 41.25% | 42.33% | 7.67 pp | -50 | 34 | -1.47 |
| BTC Daily | rf | RandomForest | 552 | 250 | 302 | 45.29% | 45.42% | 45.21% | 4.71 pp | -52 | 35 | -1.49 |
| BTC Hourly | lstm | LSTM | 755 | 345 | 410 | 45.70% | 44.58% | 48.33% | 4.30 pp | -65 | 41 | -1.59 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 161 | 219 | 42.37% | 41.67% | 42.37% | 7.63 pp | -58 | 34 | -1.71 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 159 | 221 | 41.84% | 41.25% | 41.84% | 8.16 pp | -62 | 34 | -1.82 |
| BTC Market Hours | xgb | XGBoost | 326 | 132 | 194 | 40.49% | 40.83% | 40.49% | 9.51 pp | -62 | 34 | -1.82 |
| BTC Hourly | xgb | XGBoost | 755 | 338 | 417 | 44.77% | 44.58% | 46.46% | 5.23 pp | -79 | 41 | -1.93 |
| BTC Daily | xgb | XGBoost | 552 | 227 | 325 | 41.12% | 38.33% | 40.83% | 8.88 pp | -98 | 35 | -2.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 373 | 382 | 49.40% | 46.25% | 50.62% | 0.60 pp | -9 | 41 | -0.22 |
| BTC Hourly | transformer | Transformer | 755 | 357 | 398 | 47.28% | 42.50% | 46.04% | 2.72 pp | -41 | 41 | -1.00 |
| BTC Hourly | rf | RandomForest | 755 | 356 | 399 | 47.15% | 45.83% | 47.08% | 2.85 pp | -43 | 41 | -1.05 |
| BTC Hourly | nn | NN | 755 | 354 | 401 | 46.89% | 42.50% | 46.88% | 3.11 pp | -47 | 41 | -1.15 |
| BTC Hourly | lstm | LSTM | 755 | 345 | 410 | 45.70% | 44.58% | 48.33% | 4.30 pp | -65 | 41 | -1.59 |
| BTC Hourly | xgb | XGBoost | 755 | 338 | 417 | 44.77% | 44.58% | 46.46% | 5.23 pp | -79 | 41 | -1.93 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 287 | 265 | 51.99% | 52.50% | 52.50% | 1.99 pp | 22 | 35 | 0.63 |
| BTC Daily | nn | NN | 552 | 264 | 288 | 47.83% | 45.42% | 47.50% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Daily | transformer | Transformer | 552 | 264 | 288 | 47.83% | 50.00% | 48.54% | 2.17 pp | -24 | 35 | -0.69 |
| BTC Daily | lstm | LSTM | 552 | 261 | 291 | 47.28% | 47.08% | 46.67% | 2.72 pp | -30 | 35 | -0.86 |
| BTC Daily | rf | RandomForest | 552 | 250 | 302 | 45.29% | 45.42% | 45.21% | 4.71 pp | -52 | 35 | -1.49 |
| BTC Daily | xgb | XGBoost | 552 | 227 | 325 | 41.12% | 38.33% | 40.83% | 8.88 pp | -98 | 35 | -2.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 171 | 155 | 52.45% | 51.25% | 52.45% | 2.45 pp | 16 | 34 | 0.47 |
| BTC Market Hours | nn | NN | 326 | 155 | 171 | 47.55% | 48.75% | 47.55% | 2.45 pp | -16 | 34 | -0.47 |
| BTC Market Hours | transformer | Transformer | 326 | 153 | 173 | 46.93% | 47.50% | 46.93% | 3.07 pp | -20 | 34 | -0.59 |
| BTC Market Hours | lstm | LSTM | 326 | 149 | 177 | 45.71% | 45.83% | 45.71% | 4.29 pp | -28 | 34 | -0.82 |
| BTC Market Hours | rf | RandomForest | 326 | 138 | 188 | 42.33% | 41.25% | 42.33% | 7.67 pp | -50 | 34 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 326 | 132 | 194 | 40.49% | 40.83% | 40.49% | 9.51 pp | -62 | 34 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 380 | 179 | 201 | 47.11% | 46.25% | 47.11% | 2.89 pp | -22 | 34 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 176 | 204 | 46.32% | 44.58% | 46.32% | 3.68 pp | -28 | 34 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 173 | 207 | 45.53% | 47.08% | 45.53% | 4.47 pp | -34 | 34 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 165 | 215 | 43.42% | 41.25% | 43.42% | 6.58 pp | -50 | 34 | -1.47 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 161 | 219 | 42.37% | 41.67% | 42.37% | 7.63 pp | -58 | 34 | -1.71 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 159 | 221 | 41.84% | 41.25% | 41.84% | 8.16 pp | -62 | 34 | -1.82 |

### Consolidated Hourly

_No model-level predictions available for this variation._

### Consolidated Daily/Hourly Refresh

_No model-level predictions available for this variation._

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
