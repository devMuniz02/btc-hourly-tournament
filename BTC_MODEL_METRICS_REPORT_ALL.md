# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T05:02:07.351293+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 04:00:00+00:00 | 1109 | 791 | 318 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 04:00:00+00:00 | 937 | 583 | 353 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 494 | 345 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 496 | 399 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 06:00:00+00:00 | 11 | 11 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 06:00:00+00:00 | 11 | 11 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 06:00:00+00:00 | 11 | 0 | 11 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 06:00:00+00:00 | 11 | 0 | 11 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 345 | 170 | 175 | 49.28% | 47.50% | 49.28% | 0.72 pp | -5 | 36 | -0.14 |
| BTC Daily | transformer | Transformer | 573 | 282 | 291 | 49.21% | 52.92% | 48.96% | 0.79 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 573 | 281 | 292 | 49.04% | 48.33% | 48.96% | 0.96 pp | -11 | 37 | -0.30 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 345 | 163 | 182 | 47.25% | 46.25% | 47.25% | 2.75 pp | -19 | 36 | -0.53 |
| BTC Daily | nn | NN | 573 | 271 | 302 | 47.29% | 45.83% | 48.12% | 2.71 pp | -31 | 37 | -0.84 |
| BTC Market Hours Daily | nn | NN | 399 | 184 | 215 | 46.12% | 48.33% | 46.12% | 3.88 pp | -31 | 36 | -0.86 |
| BTC Market Hours | nn | NN | 345 | 156 | 189 | 45.22% | 47.50% | 45.22% | 4.78 pp | -33 | 36 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Market Hours | lstm | LSTM | 345 | 149 | 196 | 43.19% | 43.33% | 43.19% | 6.81 pp | -47 | 36 | -1.31 |
| BTC Market Hours | rf | RandomForest | 345 | 147 | 198 | 42.61% | 42.92% | 42.61% | 7.39 pp | -51 | 36 | -1.42 |
| BTC Daily | lstm | LSTM | 573 | 258 | 315 | 45.03% | 45.83% | 44.38% | 4.97 pp | -57 | 37 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 345 | 144 | 201 | 41.74% | 42.08% | 41.74% | 8.26 pp | -57 | 36 | -1.58 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 399 | 164 | 235 | 41.10% | 39.58% | 41.10% | 8.90 pp | -71 | 36 | -1.97 |
| BTC Daily | rf | RandomForest | 573 | 248 | 325 | 43.28% | 45.00% | 43.54% | 6.72 pp | -77 | 37 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 399 | 162 | 237 | 40.60% | 40.00% | 40.60% | 9.40 pp | -75 | 36 | -2.08 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 399 | 159 | 240 | 39.85% | 37.50% | 39.85% | 10.15 pp | -81 | 36 | -2.25 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| Consolidated Hourly | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 583 | 236 | 347 | 40.48% | 35.83% | 40.83% | 9.52 pp | -111 | 37 | -3.00 |

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
| BTC Daily | transformer | Transformer | 573 | 282 | 291 | 49.21% | 52.92% | 48.96% | 0.79 pp | -9 | 37 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 573 | 281 | 292 | 49.04% | 48.33% | 48.96% | 0.96 pp | -11 | 37 | -0.30 |
| BTC Daily | nn | NN | 573 | 271 | 302 | 47.29% | 45.83% | 48.12% | 2.71 pp | -31 | 37 | -0.84 |
| BTC Daily | lstm | LSTM | 573 | 258 | 315 | 45.03% | 45.83% | 44.38% | 4.97 pp | -57 | 37 | -1.54 |
| BTC Daily | rf | RandomForest | 573 | 248 | 325 | 43.28% | 45.00% | 43.54% | 6.72 pp | -77 | 37 | -2.08 |
| BTC Daily | xgb | XGBoost | 583 | 236 | 347 | 40.48% | 35.83% | 40.83% | 9.52 pp | -111 | 37 | -3.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 345 | 170 | 175 | 49.28% | 47.50% | 49.28% | 0.72 pp | -5 | 36 | -0.14 |
| BTC Market Hours | transformer | Transformer | 345 | 163 | 182 | 47.25% | 46.25% | 47.25% | 2.75 pp | -19 | 36 | -0.53 |
| BTC Market Hours | nn | NN | 345 | 156 | 189 | 45.22% | 47.50% | 45.22% | 4.78 pp | -33 | 36 | -0.92 |
| BTC Market Hours | lstm | LSTM | 345 | 149 | 196 | 43.19% | 43.33% | 43.19% | 6.81 pp | -47 | 36 | -1.31 |
| BTC Market Hours | rf | RandomForest | 345 | 147 | 198 | 42.61% | 42.92% | 42.61% | 7.39 pp | -51 | 36 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 345 | 144 | 201 | 41.74% | 42.08% | 41.74% | 8.26 pp | -57 | 36 | -1.58 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 399 | 184 | 215 | 46.12% | 48.33% | 46.12% | 3.88 pp | -31 | 36 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 399 | 164 | 235 | 41.10% | 39.58% | 41.10% | 8.90 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | lstm | LSTM | 399 | 162 | 237 | 40.60% | 40.00% | 40.60% | 9.40 pp | -75 | 36 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 399 | 159 | 240 | 39.85% | 37.50% | 39.85% | 10.15 pp | -81 | 36 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Hourly | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 11 | 8 | 3 | 72.73% | 72.73% | 72.73% | 22.73 pp | 5 | 2 | 2.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 11 | 7 | 4 | 63.64% | 63.64% | 63.64% | 13.64 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 11 | 6 | 5 | 54.55% | 54.55% | 54.55% | 4.55 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |

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
