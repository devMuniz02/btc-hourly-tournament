# BTC Model Metrics Report - All Rows

Generated at: 2026-08-23T06:52:09.786679+00:00
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
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 346 | 170 | 176 | 49.13% | 47.50% | 49.13% | 0.87 pp | -6 | 36 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 574 | 282 | 292 | 49.13% | 48.33% | 49.17% | 0.87 pp | -10 | 37 | -0.27 |
| BTC Daily | transformer | Transformer | 574 | 281 | 293 | 48.95% | 52.08% | 48.75% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Market Hours | transformer | Transformer | 346 | 163 | 183 | 47.11% | 46.25% | 47.11% | 2.89 pp | -20 | 36 | -0.56 |
| BTC Daily | nn | NN | 574 | 271 | 303 | 47.21% | 45.83% | 48.12% | 2.79 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | nn | NN | 400 | 184 | 216 | 46.00% | 47.92% | 46.00% | 4.00 pp | -32 | 36 | -0.89 |
| BTC Market Hours | nn | NN | 346 | 157 | 189 | 45.38% | 47.50% | 45.38% | 4.62 pp | -32 | 36 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 400 | 183 | 217 | 45.75% | 46.25% | 45.75% | 4.25 pp | -34 | 36 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 400 | 182 | 218 | 45.50% | 45.42% | 45.50% | 4.50 pp | -36 | 36 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 756 | 355 | 401 | 46.96% | 43.75% | 47.50% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Hourly | transformer | Transformer | 756 | 355 | 401 | 46.96% | 44.17% | 45.83% | 3.04 pp | -46 | 42 | -1.10 |
| BTC Market Hours | lstm | LSTM | 346 | 149 | 197 | 43.06% | 43.33% | 43.06% | 6.94 pp | -48 | 36 | -1.33 |
| BTC Market Hours | rf | RandomForest | 346 | 147 | 199 | 42.49% | 42.92% | 42.49% | 7.51 pp | -52 | 36 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 346 | 145 | 201 | 41.91% | 42.50% | 41.91% | 8.09 pp | -56 | 36 | -1.56 |
| BTC Daily | lstm | LSTM | 574 | 258 | 316 | 44.95% | 45.42% | 44.38% | 5.05 pp | -58 | 37 | -1.57 |
| BTC Hourly | rf | RandomForest | 756 | 340 | 416 | 44.97% | 45.00% | 44.79% | 5.03 pp | -76 | 42 | -1.81 |
| BTC Hourly | nn | NN | 756 | 338 | 418 | 44.71% | 41.67% | 45.21% | 5.29 pp | -80 | 42 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 400 | 164 | 236 | 41.00% | 39.58% | 41.00% | 9.00 pp | -72 | 36 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 400 | 162 | 238 | 40.50% | 40.00% | 40.50% | 9.50 pp | -76 | 36 | -2.11 |
| BTC Hourly | lstm | LSTM | 756 | 333 | 423 | 44.05% | 42.92% | 45.62% | 5.95 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 574 | 247 | 327 | 43.03% | 44.17% | 43.33% | 6.97 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 400 | 160 | 240 | 40.00% | 37.92% | 40.00% | 10.00 pp | -80 | 36 | -2.22 |
| BTC Hourly | xgb | XGBoost | 756 | 326 | 430 | 43.12% | 42.08% | 44.58% | 6.88 pp | -104 | 42 | -2.48 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 584 | 235 | 349 | 40.24% | 35.42% | 40.42% | 9.76 pp | -114 | 37 | -3.08 |

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
| BTC Daily | transformer | Transformer | 574 | 281 | 293 | 48.95% | 52.08% | 48.75% | 1.05 pp | -12 | 37 | -0.32 |
| BTC Daily | nn | NN | 574 | 271 | 303 | 47.21% | 45.83% | 48.12% | 2.79 pp | -32 | 37 | -0.86 |
| BTC Daily | lstm | LSTM | 574 | 258 | 316 | 44.95% | 45.42% | 44.38% | 5.05 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 574 | 247 | 327 | 43.03% | 44.17% | 43.33% | 6.97 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 584 | 235 | 349 | 40.24% | 35.42% | 40.42% | 9.76 pp | -114 | 37 | -3.08 |

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
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

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
