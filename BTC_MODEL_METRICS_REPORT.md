# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T17:40:23.448253+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 987 | 622 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 569 | 384 | 184 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 16:00:00+00:00 | 571 | 438 | 131 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 384 | 189 | 195 | 49.22% | 47.50% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| BTC Daily | transformer | Transformer | 612 | 301 | 311 | 49.18% | 49.58% | 50.21% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 612 | 300 | 312 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Market Hours | nn | NN | 384 | 178 | 206 | 46.35% | 48.75% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 384 | 178 | 206 | 46.35% | 43.33% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| Consolidated Hourly | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 438 | 201 | 237 | 45.89% | 47.50% | 45.89% | 4.11 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 438 | 200 | 238 | 45.66% | 45.83% | 45.66% | 4.34 pp | -38 | 39 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 790 | 373 | 417 | 47.22% | 44.58% | 47.08% | 2.78 pp | -44 | 43 | -1.02 |
| BTC Daily | nn | NN | 612 | 286 | 326 | 46.73% | 43.33% | 48.33% | 3.27 pp | -40 | 38 | -1.05 |
| BTC Hourly | transformer | Transformer | 790 | 372 | 418 | 47.09% | 44.17% | 46.46% | 2.91 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 438 | 198 | 240 | 45.21% | 45.83% | 45.21% | 4.79 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 384 | 165 | 219 | 42.97% | 43.33% | 42.97% | 7.03 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 384 | 164 | 220 | 42.71% | 40.83% | 42.71% | 7.29 pp | -56 | 39 | -1.44 |
| BTC Daily | lstm | LSTM | 612 | 275 | 337 | 44.93% | 44.58% | 45.00% | 5.07 pp | -62 | 38 | -1.63 |
| BTC Hourly | nn | NN | 790 | 356 | 434 | 45.06% | 40.42% | 45.62% | 4.94 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 384 | 155 | 229 | 40.36% | 38.33% | 40.36% | 9.64 pp | -74 | 39 | -1.90 |
| BTC Hourly | rf | RandomForest | 790 | 350 | 440 | 44.30% | 42.50% | 43.75% | 5.70 pp | -90 | 43 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 438 | 178 | 260 | 40.64% | 39.17% | 40.64% | 9.36 pp | -82 | 39 | -2.10 |
| BTC Hourly | lstm | LSTM | 790 | 348 | 442 | 44.05% | 43.75% | 45.42% | 5.95 pp | -94 | 43 | -2.19 |
| BTC Daily | rf | RandomForest | 612 | 262 | 350 | 42.81% | 42.50% | 43.54% | 7.19 pp | -88 | 38 | -2.32 |
| BTC Market Hours Daily | lstm | LSTM | 438 | 173 | 265 | 39.50% | 37.50% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 438 | 173 | 265 | 39.50% | 38.33% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| BTC Hourly | xgb | XGBoost | 790 | 336 | 454 | 42.53% | 39.17% | 43.96% | 7.47 pp | -118 | 43 | -2.74 |
| Consolidated Hourly | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 622 | 247 | 375 | 39.71% | 33.33% | 39.79% | 10.29 pp | -128 | 38 | -3.37 |

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
| BTC Daily | transformer | Transformer | 612 | 301 | 311 | 49.18% | 49.58% | 50.21% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 612 | 300 | 312 | 49.02% | 47.50% | 50.21% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 612 | 286 | 326 | 46.73% | 43.33% | 48.33% | 3.27 pp | -40 | 38 | -1.05 |
| BTC Daily | lstm | LSTM | 612 | 275 | 337 | 44.93% | 44.58% | 45.00% | 5.07 pp | -62 | 38 | -1.63 |
| BTC Daily | rf | RandomForest | 612 | 262 | 350 | 42.81% | 42.50% | 43.54% | 7.19 pp | -88 | 38 | -2.32 |
| BTC Daily | xgb | XGBoost | 622 | 247 | 375 | 39.71% | 33.33% | 39.79% | 10.29 pp | -128 | 38 | -3.37 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 384 | 189 | 195 | 49.22% | 47.50% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| BTC Market Hours | nn | NN | 384 | 178 | 206 | 46.35% | 48.75% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 384 | 178 | 206 | 46.35% | 43.33% | 46.35% | 3.65 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 384 | 165 | 219 | 42.97% | 43.33% | 42.97% | 7.03 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 384 | 164 | 220 | 42.71% | 40.83% | 42.71% | 7.29 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 384 | 155 | 229 | 40.36% | 38.33% | 40.36% | 9.64 pp | -74 | 39 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 438 | 201 | 237 | 45.89% | 47.50% | 45.89% | 4.11 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 438 | 200 | 238 | 45.66% | 45.83% | 45.66% | 4.34 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 438 | 198 | 240 | 45.21% | 45.83% | 45.21% | 4.79 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 438 | 178 | 260 | 40.64% | 39.17% | 40.64% | 9.36 pp | -82 | 39 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 438 | 173 | 265 | 39.50% | 37.50% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 438 | 173 | 265 | 39.50% | 38.33% | 39.50% | 10.50 pp | -92 | 39 | -2.36 |

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
