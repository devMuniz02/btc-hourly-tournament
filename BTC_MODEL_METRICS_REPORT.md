# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T03:20:51.910709+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1138 | 850 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1013 | 648 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 616 | 410 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 618 | 464 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 66 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 06:00:00+00:00 | 66 | 0 | 66 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 410 | 204 | 206 | 49.76% | 49.17% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Daily | transformer | Transformer | 638 | 311 | 327 | 48.75% | 46.67% | 49.58% | 1.25 pp | -16 | 39 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 638 | 309 | 329 | 48.43% | 45.00% | 49.79% | 1.57 pp | -20 | 39 | -0.51 |
| BTC Market Hours | nn | NN | 410 | 194 | 216 | 47.32% | 50.42% | 47.32% | 2.68 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 410 | 190 | 220 | 46.34% | 42.50% | 46.34% | 3.66 pp | -30 | 41 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 464 | 215 | 249 | 46.34% | 46.25% | 46.34% | 3.66 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 464 | 213 | 251 | 45.91% | 46.25% | 45.91% | 4.09 pp | -38 | 41 | -0.93 |
| BTC Daily | nn | NN | 638 | 300 | 338 | 47.02% | 42.92% | 48.96% | 2.98 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 464 | 211 | 253 | 45.47% | 45.00% | 45.47% | 4.53 pp | -42 | 41 | -1.02 |
| BTC Hourly | transformer | Transformer | 816 | 385 | 431 | 47.18% | 45.83% | 46.25% | 2.82 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 816 | 383 | 433 | 46.94% | 43.33% | 47.08% | 3.06 pp | -50 | 44 | -1.14 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 410 | 181 | 229 | 44.15% | 45.42% | 44.15% | 5.85 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 410 | 176 | 234 | 42.93% | 42.08% | 42.93% | 7.07 pp | -58 | 41 | -1.41 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 816 | 368 | 448 | 45.10% | 41.25% | 45.00% | 4.90 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 638 | 283 | 355 | 44.36% | 42.08% | 43.96% | 5.64 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 816 | 365 | 451 | 44.73% | 44.58% | 44.58% | 5.27 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 464 | 190 | 274 | 40.95% | 41.25% | 40.95% | 9.05 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 41 | -2.10 |
| BTC Hourly | lstm | LSTM | 816 | 356 | 460 | 43.63% | 42.08% | 44.17% | 6.37 pp | -104 | 44 | -2.36 |
| BTC Daily | rf | RandomForest | 638 | 271 | 367 | 42.48% | 40.83% | 43.33% | 7.52 pp | -96 | 39 | -2.46 |
| BTC Market Hours Daily | xgb | XGBoost | 464 | 181 | 283 | 39.01% | 35.83% | 39.01% | 10.99 pp | -102 | 41 | -2.49 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |
| BTC Hourly | xgb | XGBoost | 816 | 345 | 471 | 42.28% | 39.58% | 42.71% | 7.72 pp | -126 | 44 | -2.86 |
| BTC Daily | xgb | XGBoost | 648 | 252 | 396 | 38.89% | 30.00% | 38.75% | 11.11 pp | -144 | 39 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 816 | 385 | 431 | 47.18% | 45.83% | 46.25% | 2.82 pp | -46 | 44 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 816 | 383 | 433 | 46.94% | 43.33% | 47.08% | 3.06 pp | -50 | 44 | -1.14 |
| BTC Hourly | nn | NN | 816 | 368 | 448 | 45.10% | 41.25% | 45.00% | 4.90 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 816 | 365 | 451 | 44.73% | 44.58% | 44.58% | 5.27 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 816 | 356 | 460 | 43.63% | 42.08% | 44.17% | 6.37 pp | -104 | 44 | -2.36 |
| BTC Hourly | xgb | XGBoost | 816 | 345 | 471 | 42.28% | 39.58% | 42.71% | 7.72 pp | -126 | 44 | -2.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 638 | 311 | 327 | 48.75% | 46.67% | 49.58% | 1.25 pp | -16 | 39 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 638 | 309 | 329 | 48.43% | 45.00% | 49.79% | 1.57 pp | -20 | 39 | -0.51 |
| BTC Daily | nn | NN | 638 | 300 | 338 | 47.02% | 42.92% | 48.96% | 2.98 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 638 | 283 | 355 | 44.36% | 42.08% | 43.96% | 5.64 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 638 | 271 | 367 | 42.48% | 40.83% | 43.33% | 7.52 pp | -96 | 39 | -2.46 |
| BTC Daily | xgb | XGBoost | 648 | 252 | 396 | 38.89% | 30.00% | 38.75% | 11.11 pp | -144 | 39 | -3.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 410 | 204 | 206 | 49.76% | 49.17% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Market Hours | nn | NN | 410 | 194 | 216 | 47.32% | 50.42% | 47.32% | 2.68 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 410 | 190 | 220 | 46.34% | 42.50% | 46.34% | 3.66 pp | -30 | 41 | -0.73 |
| BTC Market Hours | lstm | LSTM | 410 | 181 | 229 | 44.15% | 45.42% | 44.15% | 5.85 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 410 | 176 | 234 | 42.93% | 42.08% | 42.93% | 7.07 pp | -58 | 41 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 41 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 464 | 215 | 249 | 46.34% | 46.25% | 46.34% | 3.66 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 464 | 213 | 251 | 45.91% | 46.25% | 45.91% | 4.09 pp | -38 | 41 | -0.93 |
| BTC Market Hours Daily | nn | NN | 464 | 211 | 253 | 45.47% | 45.00% | 45.47% | 4.53 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 464 | 190 | 274 | 40.95% | 41.25% | 40.95% | 9.05 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 464 | 181 | 283 | 39.01% | 35.83% | 39.01% | 10.99 pp | -102 | 41 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 66 | 38 | 28 | 57.58% | 57.58% | 57.58% | 7.58 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 7 | 0.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 7 | -2.57 |

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
