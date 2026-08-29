# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T10:23:55.264701+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1125 | 837 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1000 | 635 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 590 | 397 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 592 | 451 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 06:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 06:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 06:00:00+00:00 | 55 | 0 | 55 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 06:00:00+00:00 | 55 | 0 | 55 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 397 | 195 | 202 | 49.12% | 47.50% | 49.12% | 0.88 pp | -7 | 40 | -0.17 |
| BTC Daily | transformer | Transformer | 625 | 307 | 318 | 49.12% | 47.50% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 625 | 305 | 320 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 397 | 186 | 211 | 46.85% | 49.17% | 46.85% | 3.15 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 397 | 185 | 212 | 46.60% | 42.92% | 46.60% | 3.40 pp | -27 | 40 | -0.68 |
| BTC Market Hours Daily | transformer | Transformer | 451 | 208 | 243 | 46.12% | 47.50% | 46.12% | 3.88 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 451 | 207 | 244 | 45.90% | 45.42% | 45.90% | 4.10 pp | -37 | 40 | -0.93 |
| BTC Daily | nn | NN | 625 | 293 | 332 | 46.88% | 43.33% | 48.75% | 3.12 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 451 | 205 | 246 | 45.45% | 45.83% | 45.45% | 4.55 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 803 | 379 | 424 | 47.20% | 44.58% | 46.88% | 2.80 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 803 | 378 | 425 | 47.07% | 45.00% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Market Hours | lstm | LSTM | 397 | 173 | 224 | 43.58% | 43.75% | 43.58% | 6.42 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 397 | 169 | 228 | 42.57% | 41.25% | 42.57% | 7.43 pp | -59 | 40 | -1.48 |
| Consolidated Hourly | transformer | Transformer | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 625 | 278 | 347 | 44.48% | 42.50% | 44.17% | 5.52 pp | -69 | 39 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 397 | 161 | 236 | 40.55% | 38.75% | 40.55% | 9.45 pp | -75 | 40 | -1.88 |
| BTC Hourly | nn | NN | 803 | 361 | 442 | 44.96% | 40.83% | 45.00% | 5.04 pp | -81 | 43 | -1.88 |
| BTC Hourly | rf | RandomForest | 803 | 358 | 445 | 44.58% | 44.17% | 44.17% | 5.42 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 451 | 183 | 268 | 40.58% | 39.58% | 40.58% | 9.42 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 451 | 182 | 269 | 40.35% | 38.75% | 40.35% | 9.65 pp | -87 | 40 | -2.17 |
| BTC Hourly | lstm | LSTM | 803 | 352 | 451 | 43.84% | 43.33% | 45.00% | 6.16 pp | -99 | 43 | -2.30 |
| BTC Daily | rf | RandomForest | 625 | 267 | 358 | 42.72% | 42.08% | 43.54% | 7.28 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 451 | 177 | 274 | 39.25% | 37.50% | 39.25% | 10.75 pp | -97 | 40 | -2.42 |
| BTC Hourly | xgb | XGBoost | 803 | 341 | 462 | 42.47% | 40.00% | 43.54% | 7.53 pp | -121 | 43 | -2.81 |
| Consolidated Hourly | nn | NN | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 6 | -2.83 |
| BTC Daily | xgb | XGBoost | 635 | 249 | 386 | 39.21% | 31.67% | 39.58% | 10.79 pp | -137 | 39 | -3.51 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 803 | 379 | 424 | 47.20% | 44.58% | 46.88% | 2.80 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 803 | 378 | 425 | 47.07% | 45.00% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 803 | 361 | 442 | 44.96% | 40.83% | 45.00% | 5.04 pp | -81 | 43 | -1.88 |
| BTC Hourly | rf | RandomForest | 803 | 358 | 445 | 44.58% | 44.17% | 44.17% | 5.42 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 803 | 352 | 451 | 43.84% | 43.33% | 45.00% | 6.16 pp | -99 | 43 | -2.30 |
| BTC Hourly | xgb | XGBoost | 803 | 341 | 462 | 42.47% | 40.00% | 43.54% | 7.53 pp | -121 | 43 | -2.81 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 625 | 307 | 318 | 49.12% | 47.50% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 625 | 305 | 320 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 625 | 293 | 332 | 46.88% | 43.33% | 48.75% | 3.12 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 625 | 278 | 347 | 44.48% | 42.50% | 44.17% | 5.52 pp | -69 | 39 | -1.77 |
| BTC Daily | rf | RandomForest | 625 | 267 | 358 | 42.72% | 42.08% | 43.54% | 7.28 pp | -91 | 39 | -2.33 |
| BTC Daily | xgb | XGBoost | 635 | 249 | 386 | 39.21% | 31.67% | 39.58% | 10.79 pp | -137 | 39 | -3.51 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 397 | 195 | 202 | 49.12% | 47.50% | 49.12% | 0.88 pp | -7 | 40 | -0.17 |
| BTC Market Hours | nn | NN | 397 | 186 | 211 | 46.85% | 49.17% | 46.85% | 3.15 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 397 | 185 | 212 | 46.60% | 42.92% | 46.60% | 3.40 pp | -27 | 40 | -0.68 |
| BTC Market Hours | lstm | LSTM | 397 | 173 | 224 | 43.58% | 43.75% | 43.58% | 6.42 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 397 | 169 | 228 | 42.57% | 41.25% | 42.57% | 7.43 pp | -59 | 40 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 397 | 161 | 236 | 40.55% | 38.75% | 40.55% | 9.45 pp | -75 | 40 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 451 | 208 | 243 | 46.12% | 47.50% | 46.12% | 3.88 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 451 | 207 | 244 | 45.90% | 45.42% | 45.90% | 4.10 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 451 | 205 | 246 | 45.45% | 45.83% | 45.45% | 4.55 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 451 | 183 | 268 | 40.58% | 39.58% | 40.58% | 9.42 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 451 | 182 | 269 | 40.35% | 38.75% | 40.35% | 9.65 pp | -87 | 40 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 451 | 177 | 274 | 39.25% | 37.50% | 39.25% | 10.75 pp | -97 | 40 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 6 | -2.83 |

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
