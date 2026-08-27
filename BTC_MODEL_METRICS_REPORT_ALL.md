# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T15:31:36.116939+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 804 | 315 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 968 | 603 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 14:00:00+00:00 | 535 | 365 | 169 | 1 |
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
| BTC Daily | transformer | Transformer | 593 | 294 | 299 | 49.58% | 51.67% | 50.42% | 0.42 pp | -5 | 38 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 365 | 180 | 185 | 49.32% | 48.75% | 49.32% | 0.68 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 593 | 290 | 303 | 48.90% | 47.08% | 49.38% | 1.10 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 365 | 173 | 192 | 47.40% | 46.67% | 47.40% | 2.60 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 365 | 167 | 198 | 45.75% | 47.92% | 45.75% | 4.25 pp | -31 | 37 | -0.84 |
| BTC Daily | nn | NN | 593 | 280 | 313 | 47.22% | 45.42% | 48.33% | 2.78 pp | -33 | 38 | -0.87 |
| BTC Market Hours Daily | nn | NN | 418 | 192 | 226 | 45.93% | 47.08% | 45.93% | 4.07 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 418 | 191 | 227 | 45.69% | 45.42% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 418 | 191 | 227 | 45.69% | 47.50% | 45.69% | 4.31 pp | -36 | 37 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 770 | 362 | 408 | 47.01% | 42.92% | 47.08% | 2.99 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 770 | 360 | 410 | 46.75% | 42.92% | 45.62% | 3.25 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 365 | 157 | 208 | 43.01% | 43.33% | 43.01% | 6.99 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 365 | 155 | 210 | 42.47% | 41.67% | 42.47% | 7.53 pp | -55 | 37 | -1.49 |
| BTC Daily | lstm | LSTM | 593 | 266 | 327 | 44.86% | 44.58% | 45.00% | 5.14 pp | -61 | 38 | -1.61 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 365 | 149 | 216 | 40.82% | 41.67% | 40.82% | 9.18 pp | -67 | 37 | -1.81 |
| BTC Hourly | rf | RandomForest | 770 | 345 | 425 | 44.81% | 45.00% | 44.58% | 5.19 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 770 | 343 | 427 | 44.55% | 39.58% | 45.42% | 5.45 pp | -84 | 42 | -2.00 |
| BTC Daily | rf | RandomForest | 593 | 257 | 336 | 43.34% | 44.58% | 43.75% | 6.66 pp | -79 | 38 | -2.08 |
| BTC Market Hours Daily | rf | RandomForest | 418 | 170 | 248 | 40.67% | 40.00% | 40.67% | 9.33 pp | -78 | 37 | -2.11 |
| BTC Hourly | lstm | LSTM | 770 | 340 | 430 | 44.16% | 43.33% | 45.62% | 5.84 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 418 | 166 | 252 | 39.71% | 37.92% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 418 | 166 | 252 | 39.71% | 38.33% | 39.71% | 10.29 pp | -86 | 37 | -2.32 |
| BTC Hourly | xgb | XGBoost | 770 | 330 | 440 | 42.86% | 41.25% | 44.38% | 7.14 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 603 | 243 | 360 | 40.30% | 36.25% | 40.21% | 9.70 pp | -117 | 38 | -3.08 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 770 | 362 | 408 | 47.01% | 42.92% | 47.08% | 2.99 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 770 | 360 | 410 | 46.75% | 42.92% | 45.62% | 3.25 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 770 | 345 | 425 | 44.81% | 45.00% | 44.58% | 5.19 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 770 | 343 | 427 | 44.55% | 39.58% | 45.42% | 5.45 pp | -84 | 42 | -2.00 |
| BTC Hourly | lstm | LSTM | 770 | 340 | 430 | 44.16% | 43.33% | 45.62% | 5.84 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 770 | 330 | 440 | 42.86% | 41.25% | 44.38% | 7.14 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 593 | 294 | 299 | 49.58% | 51.67% | 50.42% | 0.42 pp | -5 | 38 | -0.13 |
| BTC Daily | mlp_sklearn | MLPClassifier | 593 | 290 | 303 | 48.90% | 47.08% | 49.38% | 1.10 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 593 | 280 | 313 | 47.22% | 45.42% | 48.33% | 2.78 pp | -33 | 38 | -0.87 |
| BTC Daily | lstm | LSTM | 593 | 266 | 327 | 44.86% | 44.58% | 45.00% | 5.14 pp | -61 | 38 | -1.61 |
| BTC Daily | rf | RandomForest | 593 | 257 | 336 | 43.34% | 44.58% | 43.75% | 6.66 pp | -79 | 38 | -2.08 |
| BTC Daily | xgb | XGBoost | 603 | 243 | 360 | 40.30% | 36.25% | 40.21% | 9.70 pp | -117 | 38 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 365 | 180 | 185 | 49.32% | 48.75% | 49.32% | 0.68 pp | -5 | 37 | -0.14 |
| BTC Market Hours | transformer | Transformer | 365 | 173 | 192 | 47.40% | 46.67% | 47.40% | 2.60 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 365 | 167 | 198 | 45.75% | 47.92% | 45.75% | 4.25 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 365 | 157 | 208 | 43.01% | 43.33% | 43.01% | 6.99 pp | -51 | 37 | -1.38 |
| BTC Market Hours | rf | RandomForest | 365 | 155 | 210 | 42.47% | 41.67% | 42.47% | 7.53 pp | -55 | 37 | -1.49 |
| BTC Market Hours | xgb | XGBoost | 365 | 149 | 216 | 40.82% | 41.67% | 40.82% | 9.18 pp | -67 | 37 | -1.81 |

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
