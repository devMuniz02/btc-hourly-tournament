# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T20:02:45.550421+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1027 | 662 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 19:00:00+00:00 | 638 | 424 | 213 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 19:00:00+00:00 | 639 | 477 | 160 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 424 | 209 | 215 | 49.29% | 46.67% | 49.29% | 0.71 pp | -6 | 42 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 652 | 317 | 335 | 48.62% | 46.25% | 49.79% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 652 | 316 | 336 | 48.47% | 46.25% | 49.79% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 424 | 200 | 224 | 47.17% | 50.42% | 47.17% | 2.83 pp | -24 | 42 | -0.57 |
| Consolidated Hourly | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 477 | 220 | 257 | 46.12% | 46.67% | 46.12% | 3.88 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 424 | 193 | 231 | 45.52% | 41.67% | 45.52% | 4.48 pp | -38 | 42 | -0.90 |
| BTC Hourly | transformer | Transformer | 829 | 394 | 435 | 47.53% | 47.50% | 46.67% | 2.47 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | nn | NN | 477 | 216 | 261 | 45.28% | 44.17% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 477 | 216 | 261 | 45.28% | 45.00% | 45.28% | 4.72 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 652 | 304 | 348 | 46.63% | 41.67% | 48.75% | 3.37 pp | -44 | 40 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 829 | 388 | 441 | 46.80% | 42.08% | 46.67% | 3.20 pp | -53 | 45 | -1.18 |
| BTC Market Hours | lstm | LSTM | 424 | 185 | 239 | 43.63% | 43.33% | 43.63% | 6.37 pp | -54 | 42 | -1.29 |
| BTC Market Hours | rf | RandomForest | 424 | 183 | 241 | 43.16% | 43.33% | 43.16% | 6.84 pp | -58 | 42 | -1.38 |
| BTC Hourly | nn | NN | 829 | 375 | 454 | 45.24% | 42.92% | 44.79% | 4.76 pp | -79 | 45 | -1.76 |
| BTC Daily | lstm | LSTM | 652 | 288 | 364 | 44.17% | 41.25% | 43.54% | 5.83 pp | -76 | 40 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 477 | 197 | 280 | 41.30% | 42.08% | 41.30% | 8.70 pp | -83 | 42 | -1.98 |
| BTC Hourly | rf | RandomForest | 829 | 370 | 459 | 44.63% | 43.33% | 44.17% | 5.37 pp | -89 | 45 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 424 | 168 | 256 | 39.62% | 37.50% | 39.62% | 10.38 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 477 | 194 | 283 | 40.67% | 39.17% | 40.67% | 9.33 pp | -89 | 42 | -2.12 |
| Consolidated Hourly | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| BTC Daily | rf | RandomForest | 652 | 278 | 374 | 42.64% | 40.83% | 43.33% | 7.36 pp | -96 | 40 | -2.40 |
| BTC Hourly | lstm | LSTM | 829 | 359 | 470 | 43.31% | 40.42% | 43.33% | 6.69 pp | -111 | 45 | -2.47 |
| BTC Market Hours Daily | xgb | XGBoost | 477 | 185 | 292 | 38.78% | 35.42% | 38.78% | 11.22 pp | -107 | 42 | -2.55 |
| BTC Hourly | xgb | XGBoost | 829 | 351 | 478 | 42.34% | 39.17% | 42.50% | 7.66 pp | -127 | 45 | -2.82 |
| BTC Daily | xgb | XGBoost | 662 | 262 | 400 | 39.58% | 32.92% | 40.00% | 10.42 pp | -138 | 40 | -3.45 |

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
| BTC Daily | mlp_sklearn | MLPClassifier | 652 | 317 | 335 | 48.62% | 46.25% | 49.79% | 1.38 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 652 | 316 | 336 | 48.47% | 46.25% | 49.79% | 1.53 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 652 | 304 | 348 | 46.63% | 41.67% | 48.75% | 3.37 pp | -44 | 40 | -1.10 |
| BTC Daily | lstm | LSTM | 652 | 288 | 364 | 44.17% | 41.25% | 43.54% | 5.83 pp | -76 | 40 | -1.90 |
| BTC Daily | rf | RandomForest | 652 | 278 | 374 | 42.64% | 40.83% | 43.33% | 7.36 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 662 | 262 | 400 | 39.58% | 32.92% | 40.00% | 10.42 pp | -138 | 40 | -3.45 |

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
