# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T21:12:50.426918+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1152 | 864 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1027 | 662 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 639 | 424 | 214 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 641 | 478 | 161 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 424 | 209 | 215 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 652 | 316 | 336 | 48.47% | 45.83% | 49.58% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Daily | transformer | Transformer | 652 | 316 | 336 | 48.47% | 46.25% | 49.79% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 424 | 200 | 224 | 47.17% | 50.42% | 47.17% | 2.83 pp | -24 | 42 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 478 | 220 | 258 | 46.03% | 46.25% | 46.03% | 3.97 pp | -38 | 42 | -0.90 |
| BTC Market Hours | transformer | Transformer | 424 | 193 | 231 | 45.52% | 41.67% | 45.52% | 4.48 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 830 | 394 | 436 | 47.47% | 47.50% | 46.67% | 2.53 pp | -42 | 45 | -0.93 |
| Consolidated Hourly | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 478 | 217 | 261 | 45.40% | 44.17% | 45.40% | 4.60 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 478 | 217 | 261 | 45.40% | 45.42% | 45.40% | 4.60 pp | -44 | 42 | -1.05 |
| BTC Daily | nn | NN | 652 | 303 | 349 | 46.47% | 41.25% | 48.54% | 3.53 pp | -46 | 40 | -1.15 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 424 | 185 | 239 | 43.63% | 43.33% | 43.63% | 6.37 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 424 | 183 | 241 | 43.16% | 43.33% | 43.16% | 6.84 pp | -58 | 42 | -1.38 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Daily | lstm | LSTM | 652 | 288 | 364 | 44.17% | 41.25% | 43.54% | 5.83 pp | -76 | 40 | -1.90 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 478 | 197 | 281 | 41.21% | 41.67% | 41.21% | 8.79 pp | -84 | 42 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 424 | 168 | 256 | 39.62% | 37.50% | 39.62% | 10.38 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 478 | 194 | 284 | 40.59% | 39.17% | 40.59% | 9.41 pp | -90 | 42 | -2.14 |
| Consolidated Hourly | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 652 | 277 | 375 | 42.48% | 40.42% | 43.12% | 7.52 pp | -98 | 40 | -2.45 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Market Hours Daily | xgb | XGBoost | 478 | 185 | 293 | 38.70% | 35.42% | 38.70% | 11.30 pp | -108 | 42 | -2.57 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 662 | 261 | 401 | 39.43% | 32.50% | 39.79% | 10.57 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 830 | 394 | 436 | 47.47% | 47.50% | 46.67% | 2.53 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 652 | 316 | 336 | 48.47% | 45.83% | 49.58% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Daily | transformer | Transformer | 652 | 316 | 336 | 48.47% | 46.25% | 49.79% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 652 | 303 | 349 | 46.47% | 41.25% | 48.54% | 3.53 pp | -46 | 40 | -1.15 |
| BTC Daily | lstm | LSTM | 652 | 288 | 364 | 44.17% | 41.25% | 43.54% | 5.83 pp | -76 | 40 | -1.90 |
| BTC Daily | rf | RandomForest | 652 | 277 | 375 | 42.48% | 40.42% | 43.12% | 7.52 pp | -98 | 40 | -2.45 |
| BTC Daily | xgb | XGBoost | 662 | 261 | 401 | 39.43% | 32.50% | 39.79% | 10.57 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 424 | 209 | 215 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Market Hours | nn | NN | 424 | 200 | 224 | 47.17% | 50.42% | 47.17% | 2.83 pp | -24 | 42 | -0.57 |
| BTC Market Hours | transformer | Transformer | 424 | 193 | 231 | 45.52% | 41.67% | 45.52% | 4.48 pp | -38 | 42 | -0.90 |
| BTC Market Hours | lstm | LSTM | 424 | 185 | 239 | 43.63% | 43.33% | 43.63% | 6.37 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 424 | 183 | 241 | 43.16% | 43.33% | 43.16% | 6.84 pp | -58 | 42 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 424 | 168 | 256 | 39.62% | 37.50% | 39.62% | 10.38 pp | -88 | 42 | -2.10 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 478 | 220 | 258 | 46.03% | 46.25% | 46.03% | 3.97 pp | -38 | 42 | -0.90 |
| BTC Market Hours Daily | nn | NN | 478 | 217 | 261 | 45.40% | 44.17% | 45.40% | 4.60 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | transformer | Transformer | 478 | 217 | 261 | 45.40% | 45.42% | 45.40% | 4.60 pp | -44 | 42 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 478 | 197 | 281 | 41.21% | 41.67% | 41.21% | 8.79 pp | -84 | 42 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 478 | 194 | 284 | 40.59% | 39.17% | 40.59% | 9.41 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 478 | 185 | 293 | 38.70% | 35.42% | 38.70% | 11.30 pp | -108 | 42 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |

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
