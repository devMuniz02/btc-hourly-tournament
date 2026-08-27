# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T19:45:46.330969+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 806 | 313 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 970 | 605 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 18:00:00+00:00 | 541 | 367 | 173 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 18:00:00+00:00 | 543 | 421 | 120 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 02:00:00+00:00 | 29 | 29 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 02:00:00+00:00 | 29 | 29 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 02:00:00+00:00 | 29 | 0 | 29 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 02:00:00+00:00 | 29 | 0 | 29 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Hourly | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 367 | 181 | 186 | 49.32% | 49.17% | 49.32% | 0.68 pp | -5 | 38 | -0.13 |
| BTC Daily | transformer | Transformer | 595 | 294 | 301 | 49.41% | 50.83% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 595 | 290 | 305 | 48.74% | 46.67% | 48.96% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Market Hours | transformer | Transformer | 367 | 174 | 193 | 47.41% | 46.25% | 47.41% | 2.59 pp | -19 | 38 | -0.50 |
| BTC Market Hours | nn | NN | 367 | 169 | 198 | 46.05% | 48.33% | 46.05% | 3.95 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 595 | 280 | 315 | 47.06% | 45.00% | 48.12% | 2.94 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | nn | NN | 421 | 193 | 228 | 45.84% | 46.67% | 45.84% | 4.16 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 421 | 192 | 229 | 45.61% | 45.00% | 45.61% | 4.39 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 421 | 191 | 230 | 45.37% | 46.67% | 45.37% | 4.63 pp | -39 | 38 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 772 | 364 | 408 | 47.15% | 43.33% | 47.29% | 2.85 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 772 | 362 | 410 | 46.89% | 42.92% | 45.83% | 3.11 pp | -48 | 42 | -1.14 |
| Consolidated Hourly | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 367 | 157 | 210 | 42.78% | 43.33% | 42.78% | 7.22 pp | -53 | 38 | -1.39 |
| BTC Market Hours | rf | RandomForest | 367 | 156 | 211 | 42.51% | 41.67% | 42.51% | 7.49 pp | -55 | 38 | -1.45 |
| BTC Daily | lstm | LSTM | 595 | 266 | 329 | 44.71% | 44.17% | 44.79% | 5.29 pp | -63 | 38 | -1.66 |
| BTC Hourly | rf | RandomForest | 772 | 348 | 424 | 45.08% | 45.42% | 45.00% | 4.92 pp | -76 | 42 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 367 | 149 | 218 | 40.60% | 41.67% | 40.60% | 9.40 pp | -69 | 38 | -1.82 |
| BTC Hourly | nn | NN | 772 | 345 | 427 | 44.69% | 40.00% | 45.62% | 5.31 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 421 | 171 | 250 | 40.62% | 40.00% | 40.62% | 9.38 pp | -79 | 38 | -2.08 |
| BTC Daily | rf | RandomForest | 595 | 257 | 338 | 43.19% | 44.17% | 43.54% | 6.81 pp | -81 | 38 | -2.13 |
| BTC Hourly | lstm | LSTM | 772 | 341 | 431 | 44.17% | 42.92% | 45.62% | 5.83 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 421 | 168 | 253 | 39.90% | 38.75% | 39.90% | 10.10 pp | -85 | 38 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 421 | 166 | 255 | 39.43% | 37.92% | 39.43% | 10.57 pp | -89 | 38 | -2.34 |
| BTC Hourly | xgb | XGBoost | 772 | 332 | 440 | 43.01% | 41.67% | 44.58% | 6.99 pp | -108 | 42 | -2.57 |
| BTC Daily | xgb | XGBoost | 605 | 243 | 362 | 40.17% | 35.83% | 40.00% | 9.83 pp | -119 | 38 | -3.13 |
| Consolidated Hourly | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 772 | 364 | 408 | 47.15% | 43.33% | 47.29% | 2.85 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 772 | 362 | 410 | 46.89% | 42.92% | 45.83% | 3.11 pp | -48 | 42 | -1.14 |
| BTC Hourly | rf | RandomForest | 772 | 348 | 424 | 45.08% | 45.42% | 45.00% | 4.92 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 772 | 345 | 427 | 44.69% | 40.00% | 45.62% | 5.31 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 772 | 341 | 431 | 44.17% | 42.92% | 45.62% | 5.83 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 772 | 332 | 440 | 43.01% | 41.67% | 44.58% | 6.99 pp | -108 | 42 | -2.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 595 | 294 | 301 | 49.41% | 50.83% | 50.21% | 0.59 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 595 | 290 | 305 | 48.74% | 46.67% | 48.96% | 1.26 pp | -15 | 38 | -0.39 |
| BTC Daily | nn | NN | 595 | 280 | 315 | 47.06% | 45.00% | 48.12% | 2.94 pp | -35 | 38 | -0.92 |
| BTC Daily | lstm | LSTM | 595 | 266 | 329 | 44.71% | 44.17% | 44.79% | 5.29 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 595 | 257 | 338 | 43.19% | 44.17% | 43.54% | 6.81 pp | -81 | 38 | -2.13 |
| BTC Daily | xgb | XGBoost | 605 | 243 | 362 | 40.17% | 35.83% | 40.00% | 9.83 pp | -119 | 38 | -3.13 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 367 | 181 | 186 | 49.32% | 49.17% | 49.32% | 0.68 pp | -5 | 38 | -0.13 |
| BTC Market Hours | transformer | Transformer | 367 | 174 | 193 | 47.41% | 46.25% | 47.41% | 2.59 pp | -19 | 38 | -0.50 |
| BTC Market Hours | nn | NN | 367 | 169 | 198 | 46.05% | 48.33% | 46.05% | 3.95 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 367 | 157 | 210 | 42.78% | 43.33% | 42.78% | 7.22 pp | -53 | 38 | -1.39 |
| BTC Market Hours | rf | RandomForest | 367 | 156 | 211 | 42.51% | 41.67% | 42.51% | 7.49 pp | -55 | 38 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 367 | 149 | 218 | 40.60% | 41.67% | 40.60% | 9.40 pp | -69 | 38 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 421 | 193 | 228 | 45.84% | 46.67% | 45.84% | 4.16 pp | -35 | 38 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 421 | 192 | 229 | 45.61% | 45.00% | 45.61% | 4.39 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 421 | 191 | 230 | 45.37% | 46.67% | 45.37% | 4.63 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 421 | 171 | 250 | 40.62% | 40.00% | 40.62% | 9.38 pp | -79 | 38 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 421 | 168 | 253 | 39.90% | 38.75% | 39.90% | 10.10 pp | -85 | 38 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 421 | 166 | 255 | 39.43% | 37.92% | 39.43% | 10.57 pp | -89 | 38 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Hourly | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 29 | 18 | 11 | 62.07% | 62.07% | 62.07% | 12.07 pp | 7 | 4 | 1.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 29 | 16 | 13 | 55.17% | 55.17% | 55.17% | 5.17 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

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
