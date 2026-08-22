# BTC Model Metrics Report - All Rows

Generated at: 2026-08-22T02:50:09.843752+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-22 01:00:00+00:00 | 1081 | 790 | 291 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-22 01:00:00+00:00 | 895 | 568 | 326 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 466 | 330 | 135 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-22 00:00:00+00:00 | 467 | 383 | 82 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 3 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 3 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 0 | 3 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 09:00:00+00:00 | 3 | 0 | 3 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 330 | 164 | 166 | 49.70% | 48.75% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Daily | transformer | Transformer | 558 | 275 | 283 | 49.28% | 52.50% | 49.58% | 0.72 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 558 | 272 | 286 | 48.75% | 48.33% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Market Hours | transformer | Transformer | 330 | 155 | 175 | 46.97% | 46.67% | 46.97% | 3.03 pp | -20 | 35 | -0.57 |
| BTC Market Hours Daily | nn | NN | 383 | 176 | 207 | 45.95% | 48.33% | 45.95% | 4.05 pp | -31 | 35 | -0.89 |
| BTC Daily | nn | NN | 558 | 263 | 295 | 47.13% | 45.00% | 47.08% | 2.87 pp | -32 | 36 | -0.89 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours | nn | NN | 330 | 147 | 183 | 44.55% | 46.67% | 44.55% | 5.45 pp | -36 | 35 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 383 | 173 | 210 | 45.17% | 46.25% | 45.17% | 4.83 pp | -37 | 35 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 383 | 172 | 211 | 44.91% | 45.42% | 44.91% | 5.09 pp | -39 | 35 | -1.11 |
| BTC Market Hours | lstm | LSTM | 330 | 145 | 185 | 43.94% | 45.00% | 43.94% | 6.06 pp | -40 | 35 | -1.14 |
| BTC Daily | lstm | LSTM | 558 | 254 | 304 | 45.52% | 45.83% | 45.21% | 4.48 pp | -50 | 36 | -1.39 |
| BTC Market Hours | rf | RandomForest | 330 | 137 | 193 | 41.52% | 42.08% | 41.52% | 8.48 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 330 | 134 | 196 | 40.61% | 40.83% | 40.61% | 9.39 pp | -62 | 35 | -1.77 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Daily | rf | RandomForest | 558 | 243 | 315 | 43.55% | 45.00% | 44.17% | 6.45 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 383 | 155 | 228 | 40.47% | 38.75% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 383 | 155 | 228 | 40.47% | 37.92% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 383 | 151 | 232 | 39.43% | 37.50% | 39.43% | 10.57 pp | -81 | 35 | -2.31 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| BTC Daily | xgb | XGBoost | 568 | 230 | 338 | 40.49% | 36.25% | 40.62% | 9.51 pp | -108 | 36 | -3.00 |

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
| BTC Daily | transformer | Transformer | 558 | 275 | 283 | 49.28% | 52.50% | 49.58% | 0.72 pp | -8 | 36 | -0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 558 | 272 | 286 | 48.75% | 48.33% | 48.54% | 1.25 pp | -14 | 36 | -0.39 |
| BTC Daily | nn | NN | 558 | 263 | 295 | 47.13% | 45.00% | 47.08% | 2.87 pp | -32 | 36 | -0.89 |
| BTC Daily | lstm | LSTM | 558 | 254 | 304 | 45.52% | 45.83% | 45.21% | 4.48 pp | -50 | 36 | -1.39 |
| BTC Daily | rf | RandomForest | 558 | 243 | 315 | 43.55% | 45.00% | 44.17% | 6.45 pp | -72 | 36 | -2.00 |
| BTC Daily | xgb | XGBoost | 568 | 230 | 338 | 40.49% | 36.25% | 40.62% | 9.51 pp | -108 | 36 | -3.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 330 | 164 | 166 | 49.70% | 48.75% | 49.70% | 0.30 pp | -2 | 35 | -0.06 |
| BTC Market Hours | transformer | Transformer | 330 | 155 | 175 | 46.97% | 46.67% | 46.97% | 3.03 pp | -20 | 35 | -0.57 |
| BTC Market Hours | nn | NN | 330 | 147 | 183 | 44.55% | 46.67% | 44.55% | 5.45 pp | -36 | 35 | -1.03 |
| BTC Market Hours | lstm | LSTM | 330 | 145 | 185 | 43.94% | 45.00% | 43.94% | 6.06 pp | -40 | 35 | -1.14 |
| BTC Market Hours | rf | RandomForest | 330 | 137 | 193 | 41.52% | 42.08% | 41.52% | 8.48 pp | -56 | 35 | -1.60 |
| BTC Market Hours | xgb | XGBoost | 330 | 134 | 196 | 40.61% | 40.83% | 40.61% | 9.39 pp | -62 | 35 | -1.77 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 383 | 176 | 207 | 45.95% | 48.33% | 45.95% | 4.05 pp | -31 | 35 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 383 | 173 | 210 | 45.17% | 46.25% | 45.17% | 4.83 pp | -37 | 35 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 383 | 172 | 211 | 44.91% | 45.42% | 44.91% | 5.09 pp | -39 | 35 | -1.11 |
| BTC Market Hours Daily | lstm | LSTM | 383 | 155 | 228 | 40.47% | 38.75% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 383 | 155 | 228 | 40.47% | 37.92% | 40.47% | 9.53 pp | -73 | 35 | -2.09 |
| BTC Market Hours Daily | xgb | XGBoost | 383 | 151 | 232 | 39.43% | 37.50% | 39.43% | 10.57 pp | -81 | 35 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |

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
