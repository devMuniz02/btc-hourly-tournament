# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T08:32:19.375273+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 798 | 321 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 962 | 597 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 526 | 359 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 528 | 413 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 18:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 18:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 18:00:00+00:00 | 24 | 1 | 23 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 18:00:00+00:00 | 24 | 1 | 23 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 14 | 10 | 58.33% | 58.33% | 58.33% | 8.33 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 14 | 10 | 58.33% | 58.33% | 58.33% | 8.33 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 359 | 178 | 181 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Daily | transformer | Transformer | 587 | 291 | 296 | 49.57% | 52.08% | 50.00% | 0.43 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 587 | 287 | 300 | 48.89% | 47.50% | 49.58% | 1.11 pp | -13 | 37 | -0.35 |
| BTC Market Hours | transformer | Transformer | 359 | 170 | 189 | 47.35% | 46.25% | 47.35% | 2.65 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 359 | 164 | 195 | 45.68% | 47.92% | 45.68% | 4.32 pp | -31 | 37 | -0.84 |
| BTC Daily | nn | NN | 587 | 277 | 310 | 47.19% | 45.83% | 48.33% | 2.81 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | nn | NN | 413 | 190 | 223 | 46.00% | 47.08% | 46.00% | 4.00 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 413 | 189 | 224 | 45.76% | 46.25% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 413 | 189 | 224 | 45.76% | 47.92% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 764 | 360 | 404 | 47.12% | 43.75% | 47.71% | 2.88 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 764 | 357 | 407 | 46.73% | 43.75% | 45.62% | 3.27 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 359 | 155 | 204 | 43.18% | 42.92% | 43.18% | 6.82 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 359 | 154 | 205 | 42.90% | 42.08% | 42.90% | 7.10 pp | -51 | 37 | -1.38 |
| BTC Daily | lstm | LSTM | 587 | 264 | 323 | 44.97% | 45.00% | 45.21% | 5.03 pp | -59 | 37 | -1.59 |
| BTC Market Hours | xgb | XGBoost | 359 | 147 | 212 | 40.95% | 41.67% | 40.95% | 9.05 pp | -65 | 37 | -1.76 |
| BTC Hourly | rf | RandomForest | 764 | 342 | 422 | 44.76% | 45.00% | 44.38% | 5.24 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 764 | 341 | 423 | 44.63% | 40.42% | 45.21% | 5.37 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 413 | 169 | 244 | 40.92% | 40.00% | 40.92% | 9.08 pp | -75 | 37 | -2.03 |
| BTC Daily | rf | RandomForest | 587 | 254 | 333 | 43.27% | 43.75% | 44.17% | 6.73 pp | -79 | 37 | -2.14 |
| BTC Hourly | lstm | LSTM | 764 | 337 | 427 | 44.11% | 43.33% | 45.42% | 5.89 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 413 | 166 | 247 | 40.19% | 38.75% | 40.19% | 9.81 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 413 | 165 | 248 | 39.95% | 38.33% | 39.95% | 10.05 pp | -83 | 37 | -2.24 |
| BTC Hourly | xgb | XGBoost | 764 | 327 | 437 | 42.80% | 41.67% | 44.17% | 7.20 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 597 | 241 | 356 | 40.37% | 36.25% | 40.83% | 9.63 pp | -115 | 37 | -3.11 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 764 | 360 | 404 | 47.12% | 43.75% | 47.71% | 2.88 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 764 | 357 | 407 | 46.73% | 43.75% | 45.62% | 3.27 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 764 | 342 | 422 | 44.76% | 45.00% | 44.38% | 5.24 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 764 | 341 | 423 | 44.63% | 40.42% | 45.21% | 5.37 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 764 | 337 | 427 | 44.11% | 43.33% | 45.42% | 5.89 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 764 | 327 | 437 | 42.80% | 41.67% | 44.17% | 7.20 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 587 | 291 | 296 | 49.57% | 52.08% | 50.00% | 0.43 pp | -5 | 37 | -0.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 587 | 287 | 300 | 48.89% | 47.50% | 49.58% | 1.11 pp | -13 | 37 | -0.35 |
| BTC Daily | nn | NN | 587 | 277 | 310 | 47.19% | 45.83% | 48.33% | 2.81 pp | -33 | 37 | -0.89 |
| BTC Daily | lstm | LSTM | 587 | 264 | 323 | 44.97% | 45.00% | 45.21% | 5.03 pp | -59 | 37 | -1.59 |
| BTC Daily | rf | RandomForest | 587 | 254 | 333 | 43.27% | 43.75% | 44.17% | 6.73 pp | -79 | 37 | -2.14 |
| BTC Daily | xgb | XGBoost | 597 | 241 | 356 | 40.37% | 36.25% | 40.83% | 9.63 pp | -115 | 37 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 359 | 178 | 181 | 49.58% | 48.33% | 49.58% | 0.42 pp | -3 | 37 | -0.08 |
| BTC Market Hours | transformer | Transformer | 359 | 170 | 189 | 47.35% | 46.25% | 47.35% | 2.65 pp | -19 | 37 | -0.51 |
| BTC Market Hours | nn | NN | 359 | 164 | 195 | 45.68% | 47.92% | 45.68% | 4.32 pp | -31 | 37 | -0.84 |
| BTC Market Hours | lstm | LSTM | 359 | 155 | 204 | 43.18% | 42.92% | 43.18% | 6.82 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 359 | 154 | 205 | 42.90% | 42.08% | 42.90% | 7.10 pp | -51 | 37 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 359 | 147 | 212 | 40.95% | 41.67% | 40.95% | 9.05 pp | -65 | 37 | -1.76 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 413 | 190 | 223 | 46.00% | 47.08% | 46.00% | 4.00 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 413 | 189 | 224 | 45.76% | 46.25% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 413 | 189 | 224 | 45.76% | 47.92% | 45.76% | 4.24 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 413 | 169 | 244 | 40.92% | 40.00% | 40.92% | 9.08 pp | -75 | 37 | -2.03 |
| BTC Market Hours Daily | xgb | XGBoost | 413 | 166 | 247 | 40.19% | 38.75% | 40.19% | 9.81 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 413 | 165 | 248 | 39.95% | 38.33% | 39.95% | 10.05 pp | -83 | 37 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 14 | 10 | 58.33% | 58.33% | 58.33% | 8.33 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 14 | 10 | 58.33% | 58.33% | 58.33% | 8.33 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 12 | 12 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
