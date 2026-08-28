# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T06:01:06.753386+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 37 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 10:00:00+00:00 | 37 | 0 | 37 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 603 | 298 | 305 | 49.42% | 50.83% | 50.42% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 375 | 184 | 191 | 49.07% | 47.92% | 49.07% | 0.93 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 603 | 295 | 308 | 48.92% | 47.92% | 50.00% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 375 | 176 | 199 | 46.93% | 45.00% | 46.93% | 3.07 pp | -23 | 38 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 375 | 173 | 202 | 46.13% | 48.75% | 46.13% | 3.87 pp | -29 | 38 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 370 | 410 | 47.44% | 45.42% | 47.71% | 2.56 pp | -40 | 43 | -0.93 |
| BTC Daily | nn | NN | 603 | 283 | 320 | 46.93% | 44.58% | 47.92% | 3.07 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 429 | 195 | 234 | 45.45% | 45.42% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 429 | 195 | 234 | 45.45% | 47.08% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 429 | 195 | 234 | 45.45% | 47.92% | 45.45% | 4.55 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 780 | 366 | 414 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 43 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 375 | 160 | 215 | 42.67% | 42.92% | 42.67% | 7.33 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 375 | 159 | 216 | 42.40% | 40.00% | 42.40% | 7.60 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 603 | 269 | 334 | 44.61% | 43.33% | 44.58% | 5.39 pp | -65 | 38 | -1.71 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 780 | 350 | 430 | 44.87% | 44.17% | 44.38% | 5.13 pp | -80 | 43 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 375 | 150 | 225 | 40.00% | 39.58% | 40.00% | 10.00 pp | -75 | 38 | -1.97 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 43 | -2.05 |
| BTC Daily | rf | RandomForest | 603 | 261 | 342 | 43.28% | 44.58% | 43.75% | 6.72 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | rf | RandomForest | 429 | 174 | 255 | 40.56% | 40.42% | 40.56% | 9.44 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 429 | 169 | 260 | 39.39% | 38.33% | 39.39% | 10.61 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 780 | 336 | 444 | 43.08% | 40.83% | 44.38% | 6.92 pp | -108 | 43 | -2.51 |
| BTC Daily | xgb | XGBoost | 613 | 246 | 367 | 40.13% | 35.00% | 40.42% | 9.87 pp | -121 | 38 | -3.18 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 780 | 370 | 410 | 47.44% | 45.42% | 47.71% | 2.56 pp | -40 | 43 | -0.93 |
| BTC Hourly | transformer | Transformer | 780 | 366 | 414 | 46.92% | 42.92% | 46.04% | 3.08 pp | -48 | 43 | -1.12 |
| BTC Hourly | nn | NN | 780 | 351 | 429 | 45.00% | 40.83% | 45.83% | 5.00 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 780 | 350 | 430 | 44.87% | 44.17% | 44.38% | 5.13 pp | -80 | 43 | -1.86 |
| BTC Hourly | lstm | LSTM | 780 | 346 | 434 | 44.36% | 44.17% | 45.83% | 5.64 pp | -88 | 43 | -2.05 |
| BTC Hourly | xgb | XGBoost | 780 | 336 | 444 | 43.08% | 40.83% | 44.38% | 6.92 pp | -108 | 43 | -2.51 |

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
| Consolidated Hourly | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 37 | 21 | 16 | 56.76% | 56.76% | 56.76% | 6.76 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 37 | 20 | 17 | 54.05% | 54.05% | 54.05% | 4.05 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 37 | 15 | 22 | 40.54% | 40.54% | 40.54% | 9.46 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 4 | -3.25 |

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
