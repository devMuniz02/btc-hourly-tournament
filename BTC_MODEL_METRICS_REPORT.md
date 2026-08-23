# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T06:23:13.297455+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 05:00:00+00:00 | 1109 | 790 | 319 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 05:00:00+00:00 | 939 | 584 | 354 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 495 | 346 | 148 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 497 | 400 | 95 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 18:00:00+00:00 | 13 | 13 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 18:00:00+00:00 | 13 | 13 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 18:00:00+00:00 | 13 | 1 | 12 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 18:00:00+00:00 | 13 | 1 | 12 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 13 | 10 | 3 | 76.92% | 76.92% | 76.92% | 26.92 pp | 7 | 2 | 3.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 13 | 10 | 3 | 76.92% | 76.92% | 76.92% | 26.92 pp | 7 | 2 | 3.50 |
| Consolidated Hourly | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 346 | 170 | 176 | 49.13% | 47.50% | 49.13% | 0.87 pp | -6 | 36 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 574 | 282 | 292 | 49.13% | 48.33% | 49.17% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 574 | 282 | 292 | 49.13% | 52.50% | 48.96% | 0.87 pp | -10 | 37 | -0.27 |
| Consolidated Hourly | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 346 | 163 | 183 | 47.11% | 46.25% | 47.11% | 2.89 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 574 | 271 | 303 | 47.21% | 45.83% | 48.12% | 2.79 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 400 | 184 | 216 | 46.00% | 47.92% | 46.00% | 4.00 pp | -32 | 36 | -0.89 |
| BTC Market Hours | nn | NN | 346 | 157 | 189 | 45.38% | 47.50% | 45.38% | 4.62 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 400 | 183 | 217 | 45.75% | 46.25% | 45.75% | 4.25 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 400 | 182 | 218 | 45.50% | 45.42% | 45.50% | 4.50 pp | -36 | 36 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 346 | 149 | 197 | 43.06% | 43.33% | 43.06% | 6.94 pp | -48 | 36 | -1.33 |
| BTC Market Hours | rf | RandomForest | 346 | 147 | 199 | 42.49% | 42.92% | 42.49% | 7.51 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 346 | 145 | 201 | 41.91% | 42.50% | 41.91% | 8.09 pp | -56 | 36 | -1.56 |
| BTC Daily | lstm | LSTM | 574 | 258 | 316 | 44.95% | 45.42% | 44.38% | 5.05 pp | -58 | 37 | -1.57 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 400 | 164 | 236 | 41.00% | 39.58% | 41.00% | 9.00 pp | -72 | 36 | -2.00 |
| BTC Daily | rf | RandomForest | 574 | 248 | 326 | 43.21% | 44.58% | 43.54% | 6.79 pp | -78 | 37 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 400 | 162 | 238 | 40.50% | 40.00% | 40.50% | 9.50 pp | -76 | 36 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 400 | 160 | 240 | 40.00% | 37.92% | 40.00% | 10.00 pp | -80 | 36 | -2.22 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 2 | -2.50 |
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
| BTC Market Hours Daily | nn | NN | 400 | 184 | 216 | 46.00% | 47.92% | 46.00% | 4.00 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 400 | 183 | 217 | 45.75% | 46.25% | 45.75% | 4.25 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 400 | 182 | 218 | 45.50% | 45.42% | 45.50% | 4.50 pp | -36 | 36 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 400 | 164 | 236 | 41.00% | 39.58% | 41.00% | 9.00 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 400 | 162 | 238 | 40.50% | 40.00% | 40.50% | 9.50 pp | -76 | 36 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 400 | 160 | 240 | 40.00% | 37.92% | 40.00% | 10.00 pp | -80 | 36 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 13 | 10 | 3 | 76.92% | 76.92% | 76.92% | 26.92 pp | 7 | 2 | 3.50 |
| Consolidated Hourly | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | nn | NN | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 13 | 10 | 3 | 76.92% | 76.92% | 76.92% | 26.92 pp | 7 | 2 | 3.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 2 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
