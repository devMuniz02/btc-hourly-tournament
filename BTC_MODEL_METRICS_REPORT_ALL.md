# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T05:16:54.590566+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 814 | 305 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 978 | 613 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 555 | 375 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 557 | 429 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 36 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 0 | 36 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 09:00:00+00:00 | 36 | 0 | 36 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| BTC Daily | transformer | Transformer | 603 | 298 | 305 | 49.42% | 50.83% | 50.42% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 375 | 184 | 191 | 49.07% | 47.92% | 49.07% | 0.93 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 603 | 295 | 308 | 48.92% | 47.92% | 50.00% | 1.08 pp | -13 | 38 | -0.34 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 375 | 176 | 199 | 46.93% | 45.00% | 46.93% | 3.07 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 375 | 173 | 202 | 46.13% | 48.75% | 46.13% | 3.87 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 603 | 283 | 320 | 46.93% | 44.58% | 47.92% | 3.07 pp | -37 | 38 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 369 | 411 | 47.31% | 45.00% | 47.50% | 2.69 pp | -42 | 42 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 429 | 195 | 234 | 45.45% | 45.42% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 429 | 195 | 234 | 45.45% | 47.08% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 429 | 195 | 234 | 45.45% | 47.92% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 780 | 365 | 415 | 46.79% | 42.50% | 45.83% | 3.21 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 375 | 160 | 215 | 42.67% | 42.92% | 42.67% | 7.33 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 375 | 159 | 216 | 42.40% | 40.00% | 42.40% | 7.60 pp | -57 | 38 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| BTC Daily | lstm | LSTM | 603 | 269 | 334 | 44.61% | 43.33% | 44.58% | 5.39 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 780 | 349 | 431 | 44.74% | 43.75% | 44.17% | 5.26 pp | -82 | 42 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 375 | 150 | 225 | 40.00% | 39.58% | 40.00% | 10.00 pp | -75 | 38 | -1.97 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 42 | -2.10 |
| BTC Daily | rf | RandomForest | 603 | 261 | 342 | 43.28% | 44.58% | 43.75% | 6.72 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | rf | RandomForest | 429 | 174 | 255 | 40.56% | 40.42% | 40.56% | 9.44 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 780 | 335 | 445 | 42.95% | 40.42% | 44.17% | 7.05 pp | -110 | 42 | -2.62 |
| Consolidated Hourly | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 613 | 246 | 367 | 40.13% | 35.00% | 40.42% | 9.87 pp | -121 | 38 | -3.18 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 369 | 411 | 47.31% | 45.00% | 47.50% | 2.69 pp | -42 | 42 | -1.00 |
| BTC Hourly | transformer | Transformer | 780 | 365 | 415 | 46.79% | 42.50% | 45.83% | 3.21 pp | -50 | 42 | -1.19 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 42 | -1.86 |
| BTC Hourly | rf | RandomForest | 780 | 349 | 431 | 44.74% | 43.75% | 44.17% | 5.26 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 42 | -2.10 |
| BTC Hourly | xgb | XGBoost | 780 | 335 | 445 | 42.95% | 40.42% | 44.17% | 7.05 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 603 | 298 | 305 | 49.42% | 50.83% | 50.42% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 603 | 295 | 308 | 48.92% | 47.92% | 50.00% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 603 | 283 | 320 | 46.93% | 44.58% | 47.92% | 3.07 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 603 | 269 | 334 | 44.61% | 43.33% | 44.58% | 5.39 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 603 | 261 | 342 | 43.28% | 44.58% | 43.75% | 6.72 pp | -81 | 38 | -2.13 |
| BTC Daily | xgb | XGBoost | 613 | 246 | 367 | 40.13% | 35.00% | 40.42% | 9.87 pp | -121 | 38 | -3.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 375 | 184 | 191 | 49.07% | 47.92% | 49.07% | 0.93 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 375 | 176 | 199 | 46.93% | 45.00% | 46.93% | 3.07 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 375 | 173 | 202 | 46.13% | 48.75% | 46.13% | 3.87 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 375 | 160 | 215 | 42.67% | 42.92% | 42.67% | 7.33 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 375 | 159 | 216 | 42.40% | 40.00% | 42.40% | 7.60 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 375 | 150 | 225 | 40.00% | 39.58% | 40.00% | 10.00 pp | -75 | 38 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 429 | 195 | 234 | 45.45% | 45.42% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 429 | 195 | 234 | 45.45% | 47.08% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 429 | 195 | 234 | 45.45% | 47.92% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 429 | 174 | 255 | 40.56% | 40.42% | 40.56% | 9.44 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 36 | 20 | 16 | 55.56% | 55.56% | 55.56% | 5.56 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 36 | 15 | 21 | 41.67% | 41.67% | 41.67% | 8.33 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 4 | -3.00 |

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
