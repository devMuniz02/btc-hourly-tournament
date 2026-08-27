# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T18:23:36.131794+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 969 | 604 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 17:00:00+00:00 | 539 | 366 | 172 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 17:00:00+00:00 | 541 | 420 | 119 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 01:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 01:00:00+00:00 | 28 | 28 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 01:00:00+00:00 | 28 | 0 | 28 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 01:00:00+00:00 | 28 | 0 | 28 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 366 | 180 | 186 | 49.18% | 48.75% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Daily | transformer | Transformer | 594 | 293 | 301 | 49.33% | 50.83% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 594 | 289 | 305 | 48.65% | 46.25% | 48.96% | 1.35 pp | -16 | 38 | -0.42 |
| BTC Market Hours | transformer | Transformer | 366 | 174 | 192 | 47.54% | 46.67% | 47.54% | 2.46 pp | -18 | 38 | -0.47 |
| BTC Market Hours | nn | NN | 366 | 168 | 198 | 45.90% | 47.92% | 45.90% | 4.10 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 594 | 279 | 315 | 46.97% | 45.00% | 48.12% | 3.03 pp | -36 | 38 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 420 | 192 | 228 | 45.71% | 45.42% | 45.71% | 4.29 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | nn | NN | 420 | 192 | 228 | 45.71% | 46.67% | 45.71% | 4.29 pp | -36 | 37 | -0.97 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 420 | 191 | 229 | 45.48% | 47.08% | 45.48% | 4.52 pp | -38 | 37 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 772 | 363 | 409 | 47.02% | 42.92% | 47.08% | 2.98 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 772 | 362 | 410 | 46.89% | 42.92% | 45.83% | 3.11 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 366 | 157 | 209 | 42.90% | 43.33% | 42.90% | 7.10 pp | -52 | 38 | -1.37 |
| BTC Market Hours | rf | RandomForest | 366 | 156 | 210 | 42.62% | 42.08% | 42.62% | 7.38 pp | -54 | 38 | -1.42 |
| BTC Daily | lstm | LSTM | 594 | 266 | 328 | 44.78% | 44.17% | 45.00% | 5.22 pp | -62 | 38 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 366 | 149 | 217 | 40.71% | 41.67% | 40.71% | 9.29 pp | -68 | 38 | -1.79 |
| BTC Hourly | rf | RandomForest | 772 | 347 | 425 | 44.95% | 45.00% | 44.79% | 5.05 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 772 | 344 | 428 | 44.56% | 39.58% | 45.42% | 5.44 pp | -84 | 42 | -2.00 |
| BTC Daily | rf | RandomForest | 594 | 256 | 338 | 43.10% | 44.17% | 43.54% | 6.90 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | rf | RandomForest | 420 | 170 | 250 | 40.48% | 39.58% | 40.48% | 9.52 pp | -80 | 37 | -2.16 |
| BTC Hourly | lstm | LSTM | 772 | 340 | 432 | 44.04% | 42.50% | 45.42% | 5.96 pp | -92 | 42 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 420 | 168 | 252 | 40.00% | 38.75% | 40.00% | 10.00 pp | -84 | 37 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 420 | 166 | 254 | 39.52% | 37.92% | 39.52% | 10.48 pp | -88 | 37 | -2.38 |
| BTC Hourly | xgb | XGBoost | 772 | 332 | 440 | 43.01% | 41.67% | 44.58% | 6.99 pp | -108 | 42 | -2.57 |
| BTC Daily | xgb | XGBoost | 604 | 242 | 362 | 40.07% | 35.83% | 39.79% | 9.93 pp | -120 | 38 | -3.16 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 772 | 363 | 409 | 47.02% | 42.92% | 47.08% | 2.98 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 772 | 362 | 410 | 46.89% | 42.92% | 45.83% | 3.11 pp | -48 | 42 | -1.14 |
| BTC Hourly | rf | RandomForest | 772 | 347 | 425 | 44.95% | 45.00% | 44.79% | 5.05 pp | -78 | 42 | -1.86 |
| BTC Hourly | nn | NN | 772 | 344 | 428 | 44.56% | 39.58% | 45.42% | 5.44 pp | -84 | 42 | -2.00 |
| BTC Hourly | lstm | LSTM | 772 | 340 | 432 | 44.04% | 42.50% | 45.42% | 5.96 pp | -92 | 42 | -2.19 |
| BTC Hourly | xgb | XGBoost | 772 | 332 | 440 | 43.01% | 41.67% | 44.58% | 6.99 pp | -108 | 42 | -2.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 594 | 293 | 301 | 49.33% | 50.83% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 594 | 289 | 305 | 48.65% | 46.25% | 48.96% | 1.35 pp | -16 | 38 | -0.42 |
| BTC Daily | nn | NN | 594 | 279 | 315 | 46.97% | 45.00% | 48.12% | 3.03 pp | -36 | 38 | -0.95 |
| BTC Daily | lstm | LSTM | 594 | 266 | 328 | 44.78% | 44.17% | 45.00% | 5.22 pp | -62 | 38 | -1.63 |
| BTC Daily | rf | RandomForest | 594 | 256 | 338 | 43.10% | 44.17% | 43.54% | 6.90 pp | -82 | 38 | -2.16 |
| BTC Daily | xgb | XGBoost | 604 | 242 | 362 | 40.07% | 35.83% | 39.79% | 9.93 pp | -120 | 38 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 366 | 180 | 186 | 49.18% | 48.75% | 49.18% | 0.82 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 366 | 174 | 192 | 47.54% | 46.67% | 47.54% | 2.46 pp | -18 | 38 | -0.47 |
| BTC Market Hours | nn | NN | 366 | 168 | 198 | 45.90% | 47.92% | 45.90% | 4.10 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 366 | 157 | 209 | 42.90% | 43.33% | 42.90% | 7.10 pp | -52 | 38 | -1.37 |
| BTC Market Hours | rf | RandomForest | 366 | 156 | 210 | 42.62% | 42.08% | 42.62% | 7.38 pp | -54 | 38 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 366 | 149 | 217 | 40.71% | 41.67% | 40.71% | 9.29 pp | -68 | 38 | -1.79 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 420 | 192 | 228 | 45.71% | 45.42% | 45.71% | 4.29 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | nn | NN | 420 | 192 | 228 | 45.71% | 46.67% | 45.71% | 4.29 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 420 | 191 | 229 | 45.48% | 47.08% | 45.48% | 4.52 pp | -38 | 37 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 420 | 170 | 250 | 40.48% | 39.58% | 40.48% | 9.52 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 420 | 168 | 252 | 40.00% | 38.75% | 40.00% | 10.00 pp | -84 | 37 | -2.27 |
| BTC Market Hours Daily | lstm | LSTM | 420 | 166 | 254 | 39.52% | 37.92% | 39.52% | 10.48 pp | -88 | 37 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 28 | 17 | 11 | 60.71% | 60.71% | 60.71% | 10.71 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 28 | 16 | 12 | 57.14% | 57.14% | 57.14% | 7.14 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 28 | 14 | 14 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 28 | 7 | 21 | 25.00% | 25.00% | 25.00% | 25.00 pp | -14 | 4 | -3.50 |

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
