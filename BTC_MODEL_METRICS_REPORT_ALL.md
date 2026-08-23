# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T05:27:34.277290+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 04:00:00+00:00 | 1108 | 790 | 318 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 04:00:00+00:00 | 938 | 584 | 353 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 495 | 346 | 148 | 1 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 346 | 170 | 176 | 49.13% | 47.50% | 49.13% | 0.87 pp | -6 | 36 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 574 | 282 | 292 | 49.13% | 48.33% | 49.17% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 574 | 282 | 292 | 49.13% | 52.50% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 11 | 5 | 6 | 45.45% | 45.45% | 45.45% | 4.55 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 346 | 163 | 183 | 47.11% | 46.25% | 47.11% | 2.89 pp | -20 | 36 | -0.56 |
| BTC Market Hours Daily | nn | NN | 399 | 184 | 215 | 46.12% | 48.33% | 46.12% | 3.88 pp | -31 | 36 | -0.86 |
| BTC Daily | nn | NN | 574 | 271 | 303 | 47.21% | 45.83% | 48.12% | 2.79 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 346 | 157 | 189 | 45.38% | 47.50% | 45.38% | 4.62 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 399 | 182 | 217 | 45.61% | 45.83% | 45.61% | 4.39 pp | -35 | 36 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 346 | 149 | 197 | 43.06% | 43.33% | 43.06% | 6.94 pp | -48 | 36 | -1.33 |
| BTC Market Hours | rf | RandomForest | 346 | 147 | 199 | 42.49% | 42.92% | 42.49% | 7.51 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 346 | 145 | 201 | 41.91% | 42.50% | 41.91% | 8.09 pp | -56 | 36 | -1.56 |
| BTC Daily | lstm | LSTM | 574 | 258 | 316 | 44.95% | 45.42% | 44.38% | 5.05 pp | -58 | 37 | -1.57 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 399 | 164 | 235 | 41.10% | 39.58% | 41.10% | 8.90 pp | -71 | 36 | -1.97 |
| BTC Market Hours Daily | lstm | LSTM | 399 | 162 | 237 | 40.60% | 40.00% | 40.60% | 9.40 pp | -75 | 36 | -2.08 |
| BTC Daily | rf | RandomForest | 574 | 248 | 326 | 43.21% | 44.58% | 43.54% | 6.79 pp | -78 | 37 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 399 | 159 | 240 | 39.85% | 37.50% | 39.85% | 10.15 pp | -81 | 36 | -2.25 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 11 | 3 | 8 | 27.27% | 27.27% | 27.27% | 22.73 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 584 | 236 | 348 | 40.41% | 35.83% | 40.62% | 9.59 pp | -112 | 37 | -3.03 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 574 | 282 | 292 | 49.13% | 48.33% | 49.17% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 574 | 282 | 292 | 49.13% | 52.50% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 574 | 271 | 303 | 47.21% | 45.83% | 48.12% | 2.79 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 574 | 258 | 316 | 44.95% | 45.42% | 44.38% | 5.05 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 574 | 248 | 326 | 43.21% | 44.58% | 43.54% | 6.79 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 584 | 236 | 348 | 40.41% | 35.83% | 40.62% | 9.59 pp | -112 | 37 | -3.03 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 346 | 170 | 176 | 49.13% | 47.50% | 49.13% | 0.87 pp | -6 | 36 | -0.17 |
| BTC Market Hours | transformer | Transformer | 346 | 163 | 183 | 47.11% | 46.25% | 47.11% | 2.89 pp | -20 | 36 | -0.56 |
| BTC Market Hours | nn | NN | 346 | 157 | 189 | 45.38% | 47.50% | 45.38% | 4.62 pp | -32 | 36 | -0.89 |
| BTC Market Hours | lstm | LSTM | 346 | 149 | 197 | 43.06% | 43.33% | 43.06% | 6.94 pp | -48 | 36 | -1.33 |
| BTC Market Hours | rf | RandomForest | 346 | 147 | 199 | 42.49% | 42.92% | 42.49% | 7.51 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 346 | 145 | 201 | 41.91% | 42.50% | 41.91% | 8.09 pp | -56 | 36 | -1.56 |

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
