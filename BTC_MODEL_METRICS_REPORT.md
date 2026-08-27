# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T20:10:16.420246+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 807 | 312 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 971 | 606 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 19:00:00+00:00 | 543 | 368 | 174 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 19:00:00+00:00 | 545 | 422 | 121 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 368 | 181 | 187 | 49.18% | 49.17% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Daily | transformer | Transformer | 596 | 294 | 302 | 49.33% | 50.83% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 596 | 291 | 305 | 48.83% | 47.08% | 49.17% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Market Hours | transformer | Transformer | 368 | 174 | 194 | 47.28% | 45.83% | 47.28% | 2.72 pp | -20 | 38 | -0.53 |
| BTC Market Hours | nn | NN | 368 | 170 | 198 | 46.20% | 48.75% | 46.20% | 3.80 pp | -28 | 38 | -0.74 |
| BTC Daily | nn | NN | 596 | 281 | 315 | 47.15% | 45.42% | 48.33% | 2.85 pp | -34 | 38 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 422 | 193 | 229 | 45.73% | 45.42% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | nn | NN | 422 | 193 | 229 | 45.73% | 46.67% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 422 | 192 | 230 | 45.50% | 47.08% | 45.50% | 4.50 pp | -38 | 38 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 773 | 364 | 409 | 47.09% | 43.33% | 47.08% | 2.91 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 773 | 362 | 411 | 46.83% | 42.92% | 45.83% | 3.17 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 368 | 157 | 211 | 42.66% | 43.33% | 42.66% | 7.34 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 368 | 156 | 212 | 42.39% | 41.25% | 42.39% | 7.61 pp | -56 | 38 | -1.47 |
| BTC Daily | lstm | LSTM | 596 | 267 | 329 | 44.80% | 44.17% | 45.00% | 5.20 pp | -62 | 38 | -1.63 |
| BTC Hourly | rf | RandomForest | 773 | 348 | 425 | 45.02% | 45.00% | 44.79% | 4.98 pp | -77 | 42 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 368 | 149 | 219 | 40.49% | 41.25% | 40.49% | 9.51 pp | -70 | 38 | -1.84 |
| BTC Hourly | nn | NN | 773 | 345 | 428 | 44.63% | 39.58% | 45.42% | 5.37 pp | -83 | 42 | -1.98 |
| BTC Daily | rf | RandomForest | 596 | 258 | 338 | 43.29% | 44.58% | 43.75% | 6.71 pp | -80 | 38 | -2.11 |
| BTC Market Hours Daily | rf | RandomForest | 422 | 171 | 251 | 40.52% | 40.00% | 40.52% | 9.48 pp | -80 | 38 | -2.11 |
| BTC Hourly | lstm | LSTM | 773 | 341 | 432 | 44.11% | 42.92% | 45.62% | 5.89 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 38.75% | 39.81% | 10.19 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 422 | 166 | 256 | 39.34% | 37.92% | 39.34% | 10.66 pp | -90 | 38 | -2.37 |
| BTC Hourly | xgb | XGBoost | 773 | 332 | 441 | 42.95% | 41.25% | 44.38% | 7.05 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 606 | 244 | 362 | 40.26% | 35.83% | 40.21% | 9.74 pp | -118 | 38 | -3.11 |
| Consolidated Hourly | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 29 | 8 | 21 | 27.59% | 27.59% | 27.59% | 22.41 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 773 | 364 | 409 | 47.09% | 43.33% | 47.08% | 2.91 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 773 | 362 | 411 | 46.83% | 42.92% | 45.83% | 3.17 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 773 | 348 | 425 | 45.02% | 45.00% | 44.79% | 4.98 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 773 | 345 | 428 | 44.63% | 39.58% | 45.42% | 5.37 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 773 | 341 | 432 | 44.11% | 42.92% | 45.62% | 5.89 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 773 | 332 | 441 | 42.95% | 41.25% | 44.38% | 7.05 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 596 | 294 | 302 | 49.33% | 50.83% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 596 | 291 | 305 | 48.83% | 47.08% | 49.17% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 596 | 281 | 315 | 47.15% | 45.42% | 48.33% | 2.85 pp | -34 | 38 | -0.89 |
| BTC Daily | lstm | LSTM | 596 | 267 | 329 | 44.80% | 44.17% | 45.00% | 5.20 pp | -62 | 38 | -1.63 |
| BTC Daily | rf | RandomForest | 596 | 258 | 338 | 43.29% | 44.58% | 43.75% | 6.71 pp | -80 | 38 | -2.11 |
| BTC Daily | xgb | XGBoost | 606 | 244 | 362 | 40.26% | 35.83% | 40.21% | 9.74 pp | -118 | 38 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 368 | 181 | 187 | 49.18% | 49.17% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 368 | 174 | 194 | 47.28% | 45.83% | 47.28% | 2.72 pp | -20 | 38 | -0.53 |
| BTC Market Hours | nn | NN | 368 | 170 | 198 | 46.20% | 48.75% | 46.20% | 3.80 pp | -28 | 38 | -0.74 |
| BTC Market Hours | lstm | LSTM | 368 | 157 | 211 | 42.66% | 43.33% | 42.66% | 7.34 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 368 | 156 | 212 | 42.39% | 41.25% | 42.39% | 7.61 pp | -56 | 38 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 368 | 149 | 219 | 40.49% | 41.25% | 40.49% | 9.51 pp | -70 | 38 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 422 | 193 | 229 | 45.73% | 45.42% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | nn | NN | 422 | 193 | 229 | 45.73% | 46.67% | 45.73% | 4.27 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 422 | 192 | 230 | 45.50% | 47.08% | 45.50% | 4.50 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 422 | 171 | 251 | 40.52% | 40.00% | 40.52% | 9.48 pp | -80 | 38 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 422 | 168 | 254 | 39.81% | 38.75% | 39.81% | 10.19 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 422 | 166 | 256 | 39.34% | 37.92% | 39.34% | 10.66 pp | -90 | 38 | -2.37 |

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
