# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T05:47:29.379540+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1139 | 851 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1015 | 650 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 618 | 412 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 620 | 466 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 08:00:00+00:00 | 68 | 68 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 08:00:00+00:00 | 68 | 68 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 08:00:00+00:00 | 68 | 0 | 68 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 08:00:00+00:00 | 68 | 0 | 68 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 68 | 37 | 31 | 54.41% | 54.41% | 54.41% | 4.41 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 68 | 37 | 31 | 54.41% | 54.41% | 54.41% | 4.41 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 412 | 205 | 207 | 49.76% | 48.75% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Daily | transformer | Transformer | 640 | 312 | 328 | 48.75% | 46.67% | 49.58% | 1.25 pp | -16 | 39 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 640 | 310 | 330 | 48.44% | 45.00% | 49.79% | 1.56 pp | -20 | 39 | -0.51 |
| BTC Market Hours | nn | NN | 412 | 195 | 217 | 47.33% | 50.83% | 47.33% | 2.67 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 412 | 190 | 222 | 46.12% | 42.08% | 46.12% | 3.88 pp | -32 | 41 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 466 | 216 | 250 | 46.35% | 46.25% | 46.35% | 3.65 pp | -34 | 41 | -0.83 |
| Consolidated Hourly | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 7 | -0.86 |
| BTC Market Hours Daily | transformer | Transformer | 466 | 214 | 252 | 45.92% | 45.83% | 45.92% | 4.08 pp | -38 | 41 | -0.93 |
| BTC Daily | nn | NN | 640 | 301 | 339 | 47.03% | 42.92% | 48.96% | 2.97 pp | -38 | 39 | -0.97 |
| BTC Hourly | transformer | Transformer | 817 | 387 | 430 | 47.37% | 46.67% | 46.46% | 2.63 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | nn | NN | 466 | 212 | 254 | 45.49% | 45.00% | 45.49% | 4.51 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 817 | 385 | 432 | 47.12% | 44.17% | 47.29% | 2.88 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 412 | 182 | 230 | 44.17% | 45.00% | 44.17% | 5.83 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 412 | 177 | 235 | 42.96% | 42.08% | 42.96% | 7.04 pp | -58 | 41 | -1.41 |
| Consolidated Hourly | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| BTC Hourly | nn | NN | 817 | 370 | 447 | 45.29% | 42.08% | 45.21% | 4.71 pp | -77 | 44 | -1.75 |
| BTC Daily | lstm | LSTM | 640 | 284 | 356 | 44.38% | 42.08% | 43.96% | 5.63 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 817 | 366 | 451 | 44.80% | 45.00% | 44.58% | 5.20 pp | -85 | 44 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 466 | 192 | 274 | 41.20% | 42.08% | 41.20% | 8.80 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 466 | 191 | 275 | 40.99% | 40.42% | 40.99% | 9.01 pp | -84 | 41 | -2.05 |
| BTC Market Hours | xgb | XGBoost | 412 | 164 | 248 | 39.81% | 37.92% | 39.81% | 10.19 pp | -84 | 41 | -2.05 |
| BTC Hourly | lstm | LSTM | 817 | 356 | 461 | 43.57% | 42.08% | 44.17% | 6.43 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 640 | 272 | 368 | 42.50% | 40.83% | 43.33% | 7.50 pp | -96 | 39 | -2.46 |
| BTC Market Hours Daily | xgb | XGBoost | 466 | 181 | 285 | 38.84% | 35.83% | 38.84% | 11.16 pp | -104 | 41 | -2.54 |
| Consolidated Hourly | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |
| BTC Hourly | xgb | XGBoost | 817 | 347 | 470 | 42.47% | 40.42% | 42.92% | 7.53 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 650 | 253 | 397 | 38.92% | 30.42% | 38.96% | 11.08 pp | -144 | 39 | -3.69 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 817 | 387 | 430 | 47.37% | 46.67% | 46.46% | 2.63 pp | -43 | 44 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 817 | 385 | 432 | 47.12% | 44.17% | 47.29% | 2.88 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 817 | 370 | 447 | 45.29% | 42.08% | 45.21% | 4.71 pp | -77 | 44 | -1.75 |
| BTC Hourly | rf | RandomForest | 817 | 366 | 451 | 44.80% | 45.00% | 44.58% | 5.20 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 817 | 356 | 461 | 43.57% | 42.08% | 44.17% | 6.43 pp | -105 | 44 | -2.39 |
| BTC Hourly | xgb | XGBoost | 817 | 347 | 470 | 42.47% | 40.42% | 42.92% | 7.53 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 640 | 312 | 328 | 48.75% | 46.67% | 49.58% | 1.25 pp | -16 | 39 | -0.41 |
| BTC Daily | mlp_sklearn | MLPClassifier | 640 | 310 | 330 | 48.44% | 45.00% | 49.79% | 1.56 pp | -20 | 39 | -0.51 |
| BTC Daily | nn | NN | 640 | 301 | 339 | 47.03% | 42.92% | 48.96% | 2.97 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 640 | 284 | 356 | 44.38% | 42.08% | 43.96% | 5.63 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 640 | 272 | 368 | 42.50% | 40.83% | 43.33% | 7.50 pp | -96 | 39 | -2.46 |
| BTC Daily | xgb | XGBoost | 650 | 253 | 397 | 38.92% | 30.42% | 38.96% | 11.08 pp | -144 | 39 | -3.69 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 412 | 205 | 207 | 49.76% | 48.75% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Market Hours | nn | NN | 412 | 195 | 217 | 47.33% | 50.83% | 47.33% | 2.67 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 412 | 190 | 222 | 46.12% | 42.08% | 46.12% | 3.88 pp | -32 | 41 | -0.78 |
| BTC Market Hours | lstm | LSTM | 412 | 182 | 230 | 44.17% | 45.00% | 44.17% | 5.83 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 412 | 177 | 235 | 42.96% | 42.08% | 42.96% | 7.04 pp | -58 | 41 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 412 | 164 | 248 | 39.81% | 37.92% | 39.81% | 10.19 pp | -84 | 41 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 466 | 216 | 250 | 46.35% | 46.25% | 46.35% | 3.65 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 466 | 214 | 252 | 45.92% | 45.83% | 45.92% | 4.08 pp | -38 | 41 | -0.93 |
| BTC Market Hours Daily | nn | NN | 466 | 212 | 254 | 45.49% | 45.00% | 45.49% | 4.51 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 466 | 192 | 274 | 41.20% | 42.08% | 41.20% | 8.80 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 466 | 191 | 275 | 40.99% | 40.42% | 40.99% | 9.01 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 466 | 181 | 285 | 38.84% | 35.83% | 38.84% | 11.16 pp | -104 | 41 | -2.54 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 68 | 37 | 31 | 54.41% | 54.41% | 54.41% | 4.41 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Hourly | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 68 | 39 | 29 | 57.35% | 57.35% | 57.35% | 7.35 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 68 | 37 | 31 | 54.41% | 54.41% | 54.41% | 4.41 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 7 | -0.86 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 7 | -1.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 7 | -2.57 |

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
