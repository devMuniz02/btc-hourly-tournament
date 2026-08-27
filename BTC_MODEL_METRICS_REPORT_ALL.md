# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T04:12:06.375361+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 795 | 324 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 959 | 594 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 523 | 356 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 410 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T04:00:00+00:00 | 20 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T04:00:00+00:00 | 20 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T04:00:00+00:00 | 20 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T04:00:00+00:00 | 21 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 584 | 290 | 294 | 49.66% | 52.08% | 50.00% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 356 | 176 | 180 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 584 | 287 | 297 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| BTC Market Hours | transformer | Transformer | 356 | 168 | 188 | 47.19% | 46.25% | 47.19% | 2.81 pp | -20 | 37 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Market Hours | nn | NN | 356 | 163 | 193 | 45.79% | 47.92% | 45.79% | 4.21 pp | -30 | 37 | -0.81 |
| BTC Market Hours Daily | nn | NN | 410 | 189 | 221 | 46.10% | 47.50% | 46.10% | 3.90 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 584 | 275 | 309 | 47.09% | 45.42% | 47.92% | 2.91 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 410 | 188 | 222 | 45.85% | 46.25% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 410 | 188 | 222 | 45.85% | 47.92% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 761 | 357 | 404 | 46.91% | 43.33% | 47.50% | 3.09 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 761 | 356 | 405 | 46.78% | 43.75% | 45.62% | 3.22 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 356 | 154 | 202 | 43.26% | 42.92% | 43.26% | 6.74 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 356 | 152 | 204 | 42.70% | 41.67% | 42.70% | 7.30 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 584 | 264 | 320 | 45.21% | 46.25% | 45.21% | 4.79 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 356 | 147 | 209 | 41.29% | 42.08% | 41.29% | 8.71 pp | -62 | 37 | -1.68 |
| BTC Hourly | nn | NN | 761 | 341 | 420 | 44.81% | 41.25% | 45.62% | 5.19 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 761 | 341 | 420 | 44.81% | 45.00% | 44.58% | 5.19 pp | -79 | 42 | -1.88 |
| BTC Market Hours Daily | rf | RandomForest | 410 | 169 | 241 | 41.22% | 40.42% | 41.22% | 8.78 pp | -72 | 37 | -1.95 |
| BTC Daily | rf | RandomForest | 584 | 253 | 331 | 43.32% | 44.58% | 43.96% | 6.68 pp | -78 | 37 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 410 | 165 | 245 | 40.24% | 38.33% | 40.24% | 9.76 pp | -80 | 37 | -2.16 |
| BTC Hourly | lstm | LSTM | 761 | 335 | 426 | 44.02% | 42.92% | 45.42% | 5.98 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 37 | -2.22 |
| BTC Hourly | xgb | XGBoost | 761 | 327 | 434 | 42.97% | 42.08% | 44.38% | 7.03 pp | -107 | 42 | -2.55 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 594 | 240 | 354 | 40.40% | 36.25% | 40.83% | 9.60 pp | -114 | 37 | -3.08 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 761 | 357 | 404 | 46.91% | 43.33% | 47.50% | 3.09 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 761 | 356 | 405 | 46.78% | 43.75% | 45.62% | 3.22 pp | -49 | 42 | -1.17 |
| BTC Hourly | nn | NN | 761 | 341 | 420 | 44.81% | 41.25% | 45.62% | 5.19 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 761 | 341 | 420 | 44.81% | 45.00% | 44.58% | 5.19 pp | -79 | 42 | -1.88 |
| BTC Hourly | lstm | LSTM | 761 | 335 | 426 | 44.02% | 42.92% | 45.42% | 5.98 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 761 | 327 | 434 | 42.97% | 42.08% | 44.38% | 7.03 pp | -107 | 42 | -2.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 584 | 290 | 294 | 49.66% | 52.08% | 50.00% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 584 | 287 | 297 | 49.14% | 48.75% | 49.58% | 0.86 pp | -10 | 37 | -0.27 |
| BTC Daily | nn | NN | 584 | 275 | 309 | 47.09% | 45.42% | 47.92% | 2.91 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 584 | 264 | 320 | 45.21% | 46.25% | 45.21% | 4.79 pp | -56 | 37 | -1.51 |
| BTC Daily | rf | RandomForest | 584 | 253 | 331 | 43.32% | 44.58% | 43.96% | 6.68 pp | -78 | 37 | -2.11 |
| BTC Daily | xgb | XGBoost | 594 | 240 | 354 | 40.40% | 36.25% | 40.83% | 9.60 pp | -114 | 37 | -3.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 356 | 176 | 180 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 356 | 168 | 188 | 47.19% | 46.25% | 47.19% | 2.81 pp | -20 | 37 | -0.54 |
| BTC Market Hours | nn | NN | 356 | 163 | 193 | 45.79% | 47.92% | 45.79% | 4.21 pp | -30 | 37 | -0.81 |
| BTC Market Hours | lstm | LSTM | 356 | 154 | 202 | 43.26% | 42.92% | 43.26% | 6.74 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 356 | 152 | 204 | 42.70% | 41.67% | 42.70% | 7.30 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 356 | 147 | 209 | 41.29% | 42.08% | 41.29% | 8.71 pp | -62 | 37 | -1.68 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 410 | 189 | 221 | 46.10% | 47.50% | 46.10% | 3.90 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 410 | 188 | 222 | 45.85% | 46.25% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 410 | 188 | 222 | 45.85% | 47.92% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 410 | 169 | 241 | 41.22% | 40.42% | 41.22% | 8.78 pp | -72 | 37 | -1.95 |
| BTC Market Hours Daily | xgb | XGBoost | 410 | 165 | 245 | 40.24% | 38.33% | 40.24% | 9.76 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 37 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
