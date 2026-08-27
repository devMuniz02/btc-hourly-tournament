# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T04:54:45.334477+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 796 | 323 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 959 | 594 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 523 | 356 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 410 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 20 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 04:00:00+00:00 | 20 | 0 | 20 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 20 | 13 | 7 | 65.00% | 65.00% | 65.00% | 15.00 pp | 6 | 3 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | transformer | Transformer | 584 | 290 | 294 | 49.66% | 52.08% | 50.00% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 356 | 176 | 180 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 584 | 286 | 298 | 48.97% | 48.33% | 49.38% | 1.03 pp | -12 | 37 | -0.32 |
| BTC Market Hours | transformer | Transformer | 356 | 168 | 188 | 47.19% | 46.25% | 47.19% | 2.81 pp | -20 | 37 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Market Hours | nn | NN | 356 | 163 | 193 | 45.79% | 47.92% | 45.79% | 4.21 pp | -30 | 37 | -0.81 |
| BTC Market Hours Daily | nn | NN | 410 | 189 | 221 | 46.10% | 47.50% | 46.10% | 3.90 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 584 | 275 | 309 | 47.09% | 45.42% | 47.92% | 2.91 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 410 | 188 | 222 | 45.85% | 46.25% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 410 | 188 | 222 | 45.85% | 47.92% | 45.85% | 4.15 pp | -34 | 37 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 762 | 358 | 404 | 46.98% | 43.75% | 47.50% | 3.02 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 762 | 356 | 406 | 46.72% | 43.75% | 45.62% | 3.28 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 356 | 154 | 202 | 43.26% | 42.92% | 43.26% | 6.74 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 356 | 152 | 204 | 42.70% | 41.67% | 42.70% | 7.30 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 584 | 264 | 320 | 45.21% | 46.25% | 45.21% | 4.79 pp | -56 | 37 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 356 | 147 | 209 | 41.29% | 42.08% | 41.29% | 8.71 pp | -62 | 37 | -1.68 |
| BTC Hourly | nn | NN | 762 | 341 | 421 | 44.75% | 41.25% | 45.42% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 762 | 341 | 421 | 44.75% | 45.00% | 44.38% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 410 | 169 | 241 | 41.22% | 40.42% | 41.22% | 8.78 pp | -72 | 37 | -1.95 |
| BTC Hourly | lstm | LSTM | 762 | 336 | 426 | 44.09% | 43.33% | 45.42% | 5.91 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 584 | 252 | 332 | 43.15% | 44.17% | 43.75% | 6.85 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 410 | 165 | 245 | 40.24% | 38.33% | 40.24% | 9.76 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 37 | -2.22 |
| BTC Hourly | xgb | XGBoost | 762 | 327 | 435 | 42.91% | 42.08% | 44.17% | 7.09 pp | -108 | 42 | -2.57 |
| Consolidated Hourly | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 594 | 239 | 355 | 40.24% | 35.83% | 40.62% | 9.76 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 762 | 358 | 404 | 46.98% | 43.75% | 47.50% | 3.02 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 762 | 356 | 406 | 46.72% | 43.75% | 45.62% | 3.28 pp | -50 | 42 | -1.19 |
| BTC Hourly | nn | NN | 762 | 341 | 421 | 44.75% | 41.25% | 45.42% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 762 | 341 | 421 | 44.75% | 45.00% | 44.38% | 5.25 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 762 | 336 | 426 | 44.09% | 43.33% | 45.42% | 5.91 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 762 | 327 | 435 | 42.91% | 42.08% | 44.17% | 7.09 pp | -108 | 42 | -2.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 584 | 290 | 294 | 49.66% | 52.08% | 50.00% | 0.34 pp | -4 | 37 | -0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 584 | 286 | 298 | 48.97% | 48.33% | 49.38% | 1.03 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 584 | 275 | 309 | 47.09% | 45.42% | 47.92% | 2.91 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 584 | 264 | 320 | 45.21% | 46.25% | 45.21% | 4.79 pp | -56 | 37 | -1.51 |
| BTC Daily | rf | RandomForest | 584 | 252 | 332 | 43.15% | 44.17% | 43.75% | 6.85 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 594 | 239 | 355 | 40.24% | 35.83% | 40.62% | 9.76 pp | -116 | 37 | -3.14 |

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
