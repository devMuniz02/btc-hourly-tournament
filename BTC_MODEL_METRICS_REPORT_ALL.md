# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T05:40:31.724397+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 04:00:00+00:00 | 1085 | 791 | 294 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 04:00:00+00:00 | 899 | 569 | 329 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 467 | 331 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 469 | 385 | 82 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 331 | 164 | 167 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Daily | transformer | Transformer | 559 | 276 | 283 | 49.37% | 52.92% | 49.58% | 0.63 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 559 | 273 | 286 | 48.84% | 48.75% | 48.54% | 1.16 pp | -13 | 36 | -0.36 |
| BTC Market Hours | transformer | Transformer | 331 | 155 | 176 | 46.83% | 46.25% | 46.83% | 3.17 pp | -21 | 35 | -0.60 |
| BTC Market Hours Daily | nn | NN | 385 | 177 | 208 | 45.97% | 48.33% | 45.97% | 4.03 pp | -31 | 35 | -0.89 |
| BTC Daily | nn | NN | 559 | 263 | 296 | 47.05% | 45.00% | 47.08% | 2.95 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 385 | 175 | 210 | 45.45% | 46.67% | 45.45% | 4.55 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 385 | 174 | 211 | 45.19% | 45.83% | 45.19% | 4.81 pp | -37 | 35 | -1.06 |
| BTC Market Hours | nn | NN | 331 | 147 | 184 | 44.41% | 46.25% | 44.41% | 5.59 pp | -37 | 35 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 331 | 145 | 186 | 43.81% | 44.58% | 43.81% | 6.19 pp | -41 | 35 | -1.17 |
| BTC Daily | lstm | LSTM | 559 | 254 | 305 | 45.44% | 45.83% | 45.00% | 4.56 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 331 | 138 | 193 | 41.69% | 42.50% | 41.69% | 8.31 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 331 | 135 | 196 | 40.79% | 40.83% | 40.79% | 9.21 pp | -61 | 35 | -1.74 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 559 | 244 | 315 | 43.65% | 45.42% | 44.38% | 6.35 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | rf | RandomForest | 385 | 157 | 228 | 40.78% | 38.33% | 40.78% | 9.22 pp | -71 | 35 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 385 | 156 | 229 | 40.52% | 39.17% | 40.52% | 9.48 pp | -73 | 35 | -2.09 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 385 | 153 | 232 | 39.74% | 38.33% | 39.74% | 10.26 pp | -79 | 35 | -2.26 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| BTC Daily | xgb | XGBoost | 569 | 231 | 338 | 40.60% | 36.25% | 40.83% | 9.40 pp | -107 | 36 | -2.97 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

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
| BTC Daily | transformer | Transformer | 559 | 276 | 283 | 49.37% | 52.92% | 49.58% | 0.63 pp | -7 | 36 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 559 | 273 | 286 | 48.84% | 48.75% | 48.54% | 1.16 pp | -13 | 36 | -0.36 |
| BTC Daily | nn | NN | 559 | 263 | 296 | 47.05% | 45.00% | 47.08% | 2.95 pp | -33 | 36 | -0.92 |
| BTC Daily | lstm | LSTM | 559 | 254 | 305 | 45.44% | 45.83% | 45.00% | 4.56 pp | -51 | 36 | -1.42 |
| BTC Daily | rf | RandomForest | 559 | 244 | 315 | 43.65% | 45.42% | 44.38% | 6.35 pp | -71 | 36 | -1.97 |
| BTC Daily | xgb | XGBoost | 569 | 231 | 338 | 40.60% | 36.25% | 40.83% | 9.40 pp | -107 | 36 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 331 | 164 | 167 | 49.55% | 48.33% | 49.55% | 0.45 pp | -3 | 35 | -0.09 |
| BTC Market Hours | transformer | Transformer | 331 | 155 | 176 | 46.83% | 46.25% | 46.83% | 3.17 pp | -21 | 35 | -0.60 |
| BTC Market Hours | nn | NN | 331 | 147 | 184 | 44.41% | 46.25% | 44.41% | 5.59 pp | -37 | 35 | -1.06 |
| BTC Market Hours | lstm | LSTM | 331 | 145 | 186 | 43.81% | 44.58% | 43.81% | 6.19 pp | -41 | 35 | -1.17 |
| BTC Market Hours | rf | RandomForest | 331 | 138 | 193 | 41.69% | 42.50% | 41.69% | 8.31 pp | -55 | 35 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 331 | 135 | 196 | 40.79% | 40.83% | 40.79% | 9.21 pp | -61 | 35 | -1.74 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 385 | 177 | 208 | 45.97% | 48.33% | 45.97% | 4.03 pp | -31 | 35 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 385 | 175 | 210 | 45.45% | 46.67% | 45.45% | 4.55 pp | -35 | 35 | -1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 385 | 174 | 211 | 45.19% | 45.83% | 45.19% | 4.81 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 385 | 157 | 228 | 40.78% | 38.33% | 40.78% | 9.22 pp | -71 | 35 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 385 | 156 | 229 | 40.52% | 39.17% | 40.52% | 9.48 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 385 | 153 | 232 | 39.74% | 38.33% | 39.74% | 10.26 pp | -79 | 35 | -2.26 |

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
