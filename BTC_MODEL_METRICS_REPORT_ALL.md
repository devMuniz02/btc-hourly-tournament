# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T15:11:41.482860+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 803 | 316 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 967 | 602 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 14:00:00+00:00 | 534 | 364 | 169 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 14:00:00+00:00 | 536 | 418 | 116 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 592 | 294 | 298 | 49.66% | 52.08% | 50.42% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 364 | 179 | 185 | 49.18% | 48.33% | 49.18% | 0.82 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 592 | 290 | 302 | 48.99% | 47.50% | 49.58% | 1.01 pp | -12 | 37 | -0.32 |
| BTC Market Hours | transformer | Transformer | 364 | 172 | 192 | 47.25% | 46.25% | 47.25% | 2.75 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 364 | 166 | 198 | 45.60% | 47.50% | 45.60% | 4.40 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 592 | 279 | 313 | 47.13% | 45.00% | 48.12% | 2.87 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 418 | 192 | 226 | 45.93% | 47.08% | 45.93% | 4.07 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 418 | 191 | 227 | 45.69% | 45.42% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 418 | 191 | 227 | 45.69% | 47.50% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 769 | 362 | 407 | 47.07% | 42.92% | 47.08% | 2.93 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 769 | 360 | 409 | 46.81% | 43.33% | 45.62% | 3.19 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 364 | 156 | 208 | 42.86% | 42.92% | 42.86% | 7.14 pp | -52 | 37 | -1.41 |
| BTC Market Hours | rf | RandomForest | 364 | 154 | 210 | 42.31% | 41.25% | 42.31% | 7.69 pp | -56 | 37 | -1.51 |
| BTC Daily | lstm | LSTM | 592 | 266 | 326 | 44.93% | 44.58% | 45.00% | 5.07 pp | -60 | 37 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 364 | 148 | 216 | 40.66% | 41.25% | 40.66% | 9.34 pp | -68 | 37 | -1.84 |
| BTC Hourly | rf | RandomForest | 769 | 345 | 424 | 44.86% | 45.00% | 44.58% | 5.14 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 769 | 343 | 426 | 44.60% | 40.00% | 45.42% | 5.40 pp | -83 | 42 | -1.98 |
| BTC Daily | rf | RandomForest | 592 | 257 | 335 | 43.41% | 44.58% | 43.96% | 6.59 pp | -78 | 37 | -2.11 |
| BTC Market Hours Daily | rf | RandomForest | 418 | 170 | 248 | 40.67% | 40.00% | 40.67% | 9.33 pp | -78 | 37 | -2.11 |
| BTC Hourly | lstm | LSTM | 769 | 340 | 429 | 44.21% | 43.75% | 45.62% | 5.79 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 418 | 166 | 252 | 39.71% | 37.92% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 418 | 166 | 252 | 39.71% | 38.33% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |
| BTC Hourly | xgb | XGBoost | 769 | 330 | 439 | 42.91% | 41.25% | 44.38% | 7.09 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 602 | 242 | 360 | 40.20% | 36.25% | 40.21% | 9.80 pp | -118 | 37 | -3.19 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 769 | 362 | 407 | 47.07% | 42.92% | 47.08% | 2.93 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 769 | 360 | 409 | 46.81% | 43.33% | 45.62% | 3.19 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 769 | 345 | 424 | 44.86% | 45.00% | 44.58% | 5.14 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 769 | 343 | 426 | 44.60% | 40.00% | 45.42% | 5.40 pp | -83 | 42 | -1.98 |
| BTC Hourly | lstm | LSTM | 769 | 340 | 429 | 44.21% | 43.75% | 45.62% | 5.79 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 769 | 330 | 439 | 42.91% | 41.25% | 44.38% | 7.09 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 592 | 294 | 298 | 49.66% | 52.08% | 50.42% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 592 | 290 | 302 | 48.99% | 47.50% | 49.58% | 1.01 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 592 | 279 | 313 | 47.13% | 45.00% | 48.12% | 2.87 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 592 | 266 | 326 | 44.93% | 44.58% | 45.00% | 5.07 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 592 | 257 | 335 | 43.41% | 44.58% | 43.96% | 6.59 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 602 | 242 | 360 | 40.20% | 36.25% | 40.21% | 9.80 pp | -118 | 37 | -3.19 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 364 | 179 | 185 | 49.18% | 48.33% | 49.18% | 0.82 pp | -6 | 37 | -0.16 |
| BTC Market Hours | transformer | Transformer | 364 | 172 | 192 | 47.25% | 46.25% | 47.25% | 2.75 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 364 | 166 | 198 | 45.60% | 47.50% | 45.60% | 4.40 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 364 | 156 | 208 | 42.86% | 42.92% | 42.86% | 7.14 pp | -52 | 37 | -1.41 |
| BTC Market Hours | rf | RandomForest | 364 | 154 | 210 | 42.31% | 41.25% | 42.31% | 7.69 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 364 | 148 | 216 | 40.66% | 41.25% | 40.66% | 9.34 pp | -68 | 37 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 418 | 192 | 226 | 45.93% | 47.08% | 45.93% | 4.07 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 418 | 191 | 227 | 45.69% | 45.42% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 418 | 191 | 227 | 45.69% | 47.50% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 418 | 170 | 248 | 40.67% | 40.00% | 40.67% | 9.33 pp | -78 | 37 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 418 | 166 | 252 | 39.71% | 37.92% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 418 | 166 | 252 | 39.71% | 38.33% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

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
