# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T15:44:13.884788+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1129 | 841 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1005 | 640 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 598 | 402 | 195 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 599 | 455 | 142 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 402 | 200 | 202 | 49.75% | 48.75% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| BTC Daily | transformer | Transformer | 630 | 309 | 321 | 49.05% | 47.50% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 630 | 308 | 322 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 402 | 191 | 211 | 47.51% | 51.25% | 47.51% | 2.49 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 402 | 185 | 217 | 46.02% | 42.08% | 46.02% | 3.98 pp | -32 | 40 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 455 | 210 | 245 | 46.15% | 45.83% | 46.15% | 3.85 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 455 | 209 | 246 | 45.93% | 47.08% | 45.93% | 4.07 pp | -37 | 40 | -0.93 |
| BTC Daily | nn | NN | 630 | 296 | 334 | 46.98% | 42.92% | 48.96% | 3.02 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 455 | 207 | 248 | 45.49% | 45.83% | 45.49% | 4.51 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 402 | 178 | 224 | 44.28% | 45.83% | 44.28% | 5.72 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 402 | 173 | 229 | 43.03% | 42.08% | 43.03% | 6.97 pp | -56 | 40 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 630 | 279 | 351 | 44.29% | 42.50% | 43.75% | 5.71 pp | -72 | 39 | -1.85 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 402 | 162 | 240 | 40.30% | 38.33% | 40.30% | 9.70 pp | -78 | 40 | -1.95 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 455 | 186 | 269 | 40.88% | 40.42% | 40.88% | 9.12 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 455 | 185 | 270 | 40.66% | 39.17% | 40.66% | 9.34 pp | -85 | 40 | -2.12 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 630 | 269 | 361 | 42.70% | 42.08% | 43.54% | 7.30 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 455 | 178 | 277 | 39.12% | 37.08% | 39.12% | 10.88 pp | -99 | 40 | -2.48 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| BTC Daily | xgb | XGBoost | 640 | 251 | 389 | 39.22% | 31.67% | 39.38% | 10.78 pp | -138 | 39 | -3.54 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 630 | 309 | 321 | 49.05% | 47.50% | 49.58% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 630 | 308 | 322 | 48.89% | 46.67% | 50.21% | 1.11 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 630 | 296 | 334 | 46.98% | 42.92% | 48.96% | 3.02 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 630 | 279 | 351 | 44.29% | 42.50% | 43.75% | 5.71 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 630 | 269 | 361 | 42.70% | 42.08% | 43.54% | 7.30 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 640 | 251 | 389 | 39.22% | 31.67% | 39.38% | 10.78 pp | -138 | 39 | -3.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 402 | 200 | 202 | 49.75% | 48.75% | 49.75% | 0.25 pp | -2 | 40 | -0.05 |
| BTC Market Hours | nn | NN | 402 | 191 | 211 | 47.51% | 51.25% | 47.51% | 2.49 pp | -20 | 40 | -0.50 |
| BTC Market Hours | transformer | Transformer | 402 | 185 | 217 | 46.02% | 42.08% | 46.02% | 3.98 pp | -32 | 40 | -0.80 |
| BTC Market Hours | lstm | LSTM | 402 | 178 | 224 | 44.28% | 45.83% | 44.28% | 5.72 pp | -46 | 40 | -1.15 |
| BTC Market Hours | rf | RandomForest | 402 | 173 | 229 | 43.03% | 42.08% | 43.03% | 6.97 pp | -56 | 40 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 402 | 162 | 240 | 40.30% | 38.33% | 40.30% | 9.70 pp | -78 | 40 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 455 | 210 | 245 | 46.15% | 45.83% | 46.15% | 3.85 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 455 | 209 | 246 | 45.93% | 47.08% | 45.93% | 4.07 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 455 | 207 | 248 | 45.49% | 45.83% | 45.49% | 4.51 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 455 | 186 | 269 | 40.88% | 40.42% | 40.88% | 9.12 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 455 | 185 | 270 | 40.66% | 39.17% | 40.66% | 9.34 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 455 | 178 | 277 | 39.12% | 37.08% | 39.12% | 10.88 pp | -99 | 40 | -2.48 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

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
