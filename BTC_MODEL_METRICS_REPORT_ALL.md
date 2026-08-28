# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T17:59:52.550830+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 824 | 295 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 988 | 623 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 570 | 385 | 184 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 572 | 439 | 131 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 44 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 06:00:00+00:00 | 44 | 0 | 44 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 385 | 190 | 195 | 49.35% | 47.50% | 49.35% | 0.65 pp | -5 | 39 | -0.13 |
| BTC Daily | transformer | Transformer | 613 | 302 | 311 | 49.27% | 50.00% | 50.21% | 0.73 pp | -9 | 38 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 613 | 301 | 312 | 49.10% | 47.50% | 50.42% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Market Hours | nn | NN | 385 | 179 | 206 | 46.49% | 49.17% | 46.49% | 3.51 pp | -27 | 39 | -0.69 |
| BTC Market Hours | transformer | Transformer | 385 | 179 | 206 | 46.49% | 43.75% | 46.49% | 3.51 pp | -27 | 39 | -0.69 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 439 | 202 | 237 | 46.01% | 47.92% | 46.01% | 3.99 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 439 | 201 | 238 | 45.79% | 46.25% | 45.79% | 4.21 pp | -37 | 39 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 790 | 373 | 417 | 47.22% | 44.58% | 47.08% | 2.78 pp | -44 | 43 | -1.02 |
| BTC Daily | nn | NN | 613 | 287 | 326 | 46.82% | 43.33% | 48.54% | 3.18 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 439 | 199 | 240 | 45.33% | 46.25% | 45.33% | 4.67 pp | -41 | 39 | -1.05 |
| BTC Hourly | transformer | Transformer | 790 | 372 | 418 | 47.09% | 44.17% | 46.46% | 2.91 pp | -46 | 43 | -1.07 |
| BTC Market Hours | lstm | LSTM | 385 | 166 | 219 | 43.12% | 43.75% | 43.12% | 6.88 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 385 | 165 | 220 | 42.86% | 41.25% | 42.86% | 7.14 pp | -55 | 39 | -1.41 |
| BTC Daily | lstm | LSTM | 613 | 275 | 338 | 44.86% | 44.17% | 44.79% | 5.14 pp | -63 | 38 | -1.66 |
| BTC Hourly | nn | NN | 790 | 356 | 434 | 45.06% | 40.42% | 45.62% | 4.94 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 385 | 156 | 229 | 40.52% | 38.75% | 40.52% | 9.48 pp | -73 | 39 | -1.87 |
| BTC Market Hours Daily | rf | RandomForest | 439 | 179 | 260 | 40.77% | 39.58% | 40.77% | 9.23 pp | -81 | 39 | -2.08 |
| BTC Hourly | rf | RandomForest | 790 | 350 | 440 | 44.30% | 42.50% | 43.75% | 5.70 pp | -90 | 43 | -2.09 |
| BTC Hourly | lstm | LSTM | 790 | 348 | 442 | 44.05% | 43.75% | 45.42% | 5.95 pp | -94 | 43 | -2.19 |
| BTC Daily | rf | RandomForest | 613 | 263 | 350 | 42.90% | 42.50% | 43.75% | 7.10 pp | -87 | 38 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 439 | 174 | 265 | 39.64% | 37.50% | 39.64% | 10.36 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 439 | 173 | 266 | 39.41% | 37.92% | 39.41% | 10.59 pp | -93 | 39 | -2.38 |
| BTC Hourly | xgb | XGBoost | 790 | 336 | 454 | 42.53% | 39.17% | 43.96% | 7.47 pp | -118 | 43 | -2.74 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 623 | 248 | 375 | 39.81% | 33.33% | 40.00% | 10.19 pp | -127 | 38 | -3.34 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 790 | 373 | 417 | 47.22% | 44.58% | 47.08% | 2.78 pp | -44 | 43 | -1.02 |
| BTC Hourly | transformer | Transformer | 790 | 372 | 418 | 47.09% | 44.17% | 46.46% | 2.91 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 790 | 356 | 434 | 45.06% | 40.42% | 45.62% | 4.94 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 790 | 350 | 440 | 44.30% | 42.50% | 43.75% | 5.70 pp | -90 | 43 | -2.09 |
| BTC Hourly | lstm | LSTM | 790 | 348 | 442 | 44.05% | 43.75% | 45.42% | 5.95 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 790 | 336 | 454 | 42.53% | 39.17% | 43.96% | 7.47 pp | -118 | 43 | -2.74 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 613 | 302 | 311 | 49.27% | 50.00% | 50.21% | 0.73 pp | -9 | 38 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 613 | 301 | 312 | 49.10% | 47.50% | 50.42% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | nn | NN | 613 | 287 | 326 | 46.82% | 43.33% | 48.54% | 3.18 pp | -39 | 38 | -1.03 |
| BTC Daily | lstm | LSTM | 613 | 275 | 338 | 44.86% | 44.17% | 44.79% | 5.14 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 613 | 263 | 350 | 42.90% | 42.50% | 43.75% | 7.10 pp | -87 | 38 | -2.29 |
| BTC Daily | xgb | XGBoost | 623 | 248 | 375 | 39.81% | 33.33% | 40.00% | 10.19 pp | -127 | 38 | -3.34 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 385 | 190 | 195 | 49.35% | 47.50% | 49.35% | 0.65 pp | -5 | 39 | -0.13 |
| BTC Market Hours | nn | NN | 385 | 179 | 206 | 46.49% | 49.17% | 46.49% | 3.51 pp | -27 | 39 | -0.69 |
| BTC Market Hours | transformer | Transformer | 385 | 179 | 206 | 46.49% | 43.75% | 46.49% | 3.51 pp | -27 | 39 | -0.69 |
| BTC Market Hours | lstm | LSTM | 385 | 166 | 219 | 43.12% | 43.75% | 43.12% | 6.88 pp | -53 | 39 | -1.36 |
| BTC Market Hours | rf | RandomForest | 385 | 165 | 220 | 42.86% | 41.25% | 42.86% | 7.14 pp | -55 | 39 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 385 | 156 | 229 | 40.52% | 38.75% | 40.52% | 9.48 pp | -73 | 39 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 439 | 202 | 237 | 46.01% | 47.92% | 46.01% | 3.99 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 439 | 201 | 238 | 45.79% | 46.25% | 45.79% | 4.21 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | nn | NN | 439 | 199 | 240 | 45.33% | 46.25% | 45.33% | 4.67 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 439 | 179 | 260 | 40.77% | 39.58% | 40.77% | 9.23 pp | -81 | 39 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 439 | 174 | 265 | 39.64% | 37.50% | 39.64% | 10.36 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 439 | 173 | 266 | 39.41% | 37.92% | 39.41% | 10.59 pp | -93 | 39 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 44 | 25 | 19 | 56.82% | 56.82% | 56.82% | 6.82 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |

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
