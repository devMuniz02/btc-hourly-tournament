# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T15:50:43.822790+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 14:00:00+00:00 | 1095 | 791 | 304 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 14:00:00+00:00 | 915 | 575 | 339 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 14:00:00+00:00 | 476 | 337 | 138 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 14:00:00+00:00 | 478 | 391 | 85 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 5 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 11:00:00+00:00 | 5 | 0 | 5 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 337 | 167 | 170 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 565 | 279 | 286 | 49.38% | 52.92% | 49.38% | 0.62 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 565 | 277 | 288 | 49.03% | 48.33% | 49.17% | 0.97 pp | -11 | 36 | -0.31 |
| BTC Market Hours | transformer | Transformer | 337 | 158 | 179 | 46.88% | 46.25% | 46.88% | 3.12 pp | -21 | 35 | -0.60 |
| BTC Market Hours Daily | nn | NN | 391 | 182 | 209 | 46.55% | 48.75% | 46.55% | 3.45 pp | -27 | 35 | -0.77 |
| BTC Daily | nn | NN | 565 | 268 | 297 | 47.43% | 45.83% | 47.92% | 2.57 pp | -29 | 36 | -0.81 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 391 | 178 | 213 | 45.52% | 45.00% | 45.52% | 4.48 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 391 | 178 | 213 | 45.52% | 45.83% | 45.52% | 4.48 pp | -35 | 35 | -1.00 |
| BTC Market Hours | nn | NN | 337 | 151 | 186 | 44.81% | 46.67% | 44.81% | 5.19 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 337 | 147 | 190 | 43.62% | 44.17% | 43.62% | 6.38 pp | -43 | 35 | -1.23 |
| BTC Daily | lstm | LSTM | 565 | 255 | 310 | 45.13% | 45.83% | 44.79% | 4.87 pp | -55 | 36 | -1.53 |
| BTC Market Hours | rf | RandomForest | 337 | 140 | 197 | 41.54% | 41.67% | 41.54% | 8.46 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 337 | 137 | 200 | 40.65% | 40.00% | 40.65% | 9.35 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 565 | 246 | 319 | 43.54% | 45.42% | 44.38% | 6.46 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | rf | RandomForest | 391 | 159 | 232 | 40.66% | 37.92% | 40.66% | 9.34 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 391 | 158 | 233 | 40.41% | 38.75% | 40.41% | 9.59 pp | -75 | 35 | -2.14 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 391 | 153 | 238 | 39.13% | 36.25% | 39.13% | 10.87 pp | -85 | 35 | -2.43 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | xgb | XGBoost | 575 | 233 | 342 | 40.52% | 36.25% | 40.42% | 9.48 pp | -109 | 36 | -3.03 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 565 | 279 | 286 | 49.38% | 52.92% | 49.38% | 0.62 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 565 | 277 | 288 | 49.03% | 48.33% | 49.17% | 0.97 pp | -11 | 36 | -0.31 |
| BTC Daily | nn | NN | 565 | 268 | 297 | 47.43% | 45.83% | 47.92% | 2.57 pp | -29 | 36 | -0.81 |
| BTC Daily | lstm | LSTM | 565 | 255 | 310 | 45.13% | 45.83% | 44.79% | 4.87 pp | -55 | 36 | -1.53 |
| BTC Daily | rf | RandomForest | 565 | 246 | 319 | 43.54% | 45.42% | 44.38% | 6.46 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 575 | 233 | 342 | 40.52% | 36.25% | 40.42% | 9.48 pp | -109 | 36 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 337 | 167 | 170 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 337 | 158 | 179 | 46.88% | 46.25% | 46.88% | 3.12 pp | -21 | 35 | -0.60 |
| BTC Market Hours | nn | NN | 337 | 151 | 186 | 44.81% | 46.67% | 44.81% | 5.19 pp | -35 | 35 | -1.00 |
| BTC Market Hours | lstm | LSTM | 337 | 147 | 190 | 43.62% | 44.17% | 43.62% | 6.38 pp | -43 | 35 | -1.23 |
| BTC Market Hours | rf | RandomForest | 337 | 140 | 197 | 41.54% | 41.67% | 41.54% | 8.46 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 337 | 137 | 200 | 40.65% | 40.00% | 40.65% | 9.35 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 391 | 182 | 209 | 46.55% | 48.75% | 46.55% | 3.45 pp | -27 | 35 | -0.77 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 391 | 178 | 213 | 45.52% | 45.00% | 45.52% | 4.48 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 391 | 178 | 213 | 45.52% | 45.83% | 45.52% | 4.48 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 391 | 159 | 232 | 40.66% | 37.92% | 40.66% | 9.34 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 391 | 158 | 233 | 40.41% | 38.75% | 40.41% | 9.59 pp | -75 | 35 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 391 | 153 | 238 | 39.13% | 36.25% | 39.13% | 10.87 pp | -85 | 35 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 5 | 3 | 2 | 60.00% | 60.00% | 60.00% | 10.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

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
