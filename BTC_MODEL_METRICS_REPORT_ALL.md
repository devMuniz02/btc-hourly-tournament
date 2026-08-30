# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T19:53:27.836090+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1151 | 863 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1026 | 661 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 18:00:00+00:00 | 636 | 423 | 212 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 18:00:00+00:00 | 638 | 477 | 159 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 77 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 77 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 0 | 77 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 0 | 77 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 423 | 208 | 215 | 49.17% | 46.25% | 49.17% | 0.83 pp | -7 | 42 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 651 | 316 | 335 | 48.54% | 45.83% | 49.58% | 1.46 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 651 | 315 | 336 | 48.39% | 45.83% | 49.58% | 1.61 pp | -21 | 40 | -0.53 |
| BTC Market Hours | nn | NN | 423 | 199 | 224 | 47.04% | 50.42% | 47.04% | 2.96 pp | -25 | 42 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 477 | 220 | 257 | 46.12% | 46.67% | 46.12% | 3.88 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 423 | 193 | 230 | 45.63% | 41.67% | 45.63% | 4.37 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 829 | 394 | 435 | 47.53% | 47.50% | 46.67% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 477 | 216 | 261 | 45.28% | 44.17% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 477 | 216 | 261 | 45.28% | 45.00% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 651 | 303 | 348 | 46.54% | 41.67% | 48.54% | 3.46 pp | -45 | 40 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 829 | 388 | 441 | 46.80% | 42.08% | 46.67% | 3.20 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 423 | 185 | 238 | 43.74% | 43.33% | 43.74% | 6.26 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 423 | 182 | 241 | 43.03% | 42.92% | 43.03% | 6.97 pp | -59 | 42 | -1.40 |
| BTC Hourly | nn | NN | 829 | 375 | 454 | 45.24% | 42.92% | 44.79% | 4.76 pp | -79 | 45 | -1.76 |
| BTC Daily | lstm | LSTM | 651 | 288 | 363 | 44.24% | 41.67% | 43.54% | 5.76 pp | -75 | 40 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 477 | 197 | 280 | 41.30% | 42.08% | 41.30% | 8.70 pp | -83 | 42 | -1.98 |
| BTC Hourly | rf | RandomForest | 829 | 370 | 459 | 44.63% | 43.33% | 44.17% | 5.37 pp | -89 | 45 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 37.50% | 39.72% | 10.28 pp | -87 | 42 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 477 | 194 | 283 | 40.67% | 39.17% | 40.67% | 9.33 pp | -89 | 42 | -2.12 |
| Consolidated Hourly | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| BTC Daily | rf | RandomForest | 651 | 277 | 374 | 42.55% | 40.42% | 43.12% | 7.45 pp | -97 | 40 | -2.42 |
| BTC Hourly | lstm | LSTM | 829 | 359 | 470 | 43.31% | 40.42% | 43.33% | 6.69 pp | -111 | 45 | -2.47 |
| BTC Market Hours Daily | xgb | XGBoost | 477 | 185 | 292 | 38.78% | 35.42% | 38.78% | 11.22 pp | -107 | 42 | -2.55 |
| BTC Hourly | xgb | XGBoost | 829 | 351 | 478 | 42.34% | 39.17% | 42.50% | 7.66 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 661 | 261 | 400 | 39.49% | 32.50% | 40.00% | 10.51 pp | -139 | 40 | -3.48 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 829 | 394 | 435 | 47.53% | 47.50% | 46.67% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 829 | 388 | 441 | 46.80% | 42.08% | 46.67% | 3.20 pp | -53 | 45 | -1.18 |
| BTC Hourly | nn | NN | 829 | 375 | 454 | 45.24% | 42.92% | 44.79% | 4.76 pp | -79 | 45 | -1.76 |
| BTC Hourly | rf | RandomForest | 829 | 370 | 459 | 44.63% | 43.33% | 44.17% | 5.37 pp | -89 | 45 | -1.98 |
| BTC Hourly | lstm | LSTM | 829 | 359 | 470 | 43.31% | 40.42% | 43.33% | 6.69 pp | -111 | 45 | -2.47 |
| BTC Hourly | xgb | XGBoost | 829 | 351 | 478 | 42.34% | 39.17% | 42.50% | 7.66 pp | -127 | 45 | -2.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 651 | 316 | 335 | 48.54% | 45.83% | 49.58% | 1.46 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 651 | 315 | 336 | 48.39% | 45.83% | 49.58% | 1.61 pp | -21 | 40 | -0.53 |
| BTC Daily | nn | NN | 651 | 303 | 348 | 46.54% | 41.67% | 48.54% | 3.46 pp | -45 | 40 | -1.12 |
| BTC Daily | lstm | LSTM | 651 | 288 | 363 | 44.24% | 41.67% | 43.54% | 5.76 pp | -75 | 40 | -1.88 |
| BTC Daily | rf | RandomForest | 651 | 277 | 374 | 42.55% | 40.42% | 43.12% | 7.45 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 661 | 261 | 400 | 39.49% | 32.50% | 40.00% | 10.51 pp | -139 | 40 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 423 | 208 | 215 | 49.17% | 46.25% | 49.17% | 0.83 pp | -7 | 42 | -0.17 |
| BTC Market Hours | nn | NN | 423 | 199 | 224 | 47.04% | 50.42% | 47.04% | 2.96 pp | -25 | 42 | -0.60 |
| BTC Market Hours | transformer | Transformer | 423 | 193 | 230 | 45.63% | 41.67% | 45.63% | 4.37 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 423 | 185 | 238 | 43.74% | 43.33% | 43.74% | 6.26 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 423 | 182 | 241 | 43.03% | 42.92% | 43.03% | 6.97 pp | -59 | 42 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 423 | 168 | 255 | 39.72% | 37.50% | 39.72% | 10.28 pp | -87 | 42 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 477 | 220 | 257 | 46.12% | 46.67% | 46.12% | 3.88 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | nn | NN | 477 | 216 | 261 | 45.28% | 44.17% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 477 | 216 | 261 | 45.28% | 45.00% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 477 | 197 | 280 | 41.30% | 42.08% | 41.30% | 8.70 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 477 | 194 | 283 | 40.67% | 39.17% | 40.67% | 9.33 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 477 | 185 | 292 | 38.78% | 35.42% | 38.78% | 11.22 pp | -107 | 42 | -2.55 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |

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
