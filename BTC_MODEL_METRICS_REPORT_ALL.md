# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T10:22:01.905547+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 800 | 319 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 964 | 599 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 528 | 361 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 529 | 414 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 0 | 24 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 0 | 24 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 361 | 178 | 183 | 49.31% | 47.92% | 49.31% | 0.69 pp | -5 | 37 | -0.14 |
| BTC Daily | transformer | Transformer | 589 | 291 | 298 | 49.41% | 51.67% | 49.79% | 0.59 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 589 | 287 | 302 | 48.73% | 47.08% | 49.38% | 1.27 pp | -15 | 37 | -0.41 |
| BTC Market Hours | transformer | Transformer | 361 | 172 | 189 | 47.65% | 46.25% | 47.65% | 2.35 pp | -17 | 37 | -0.46 |
| Consolidated Hourly | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 414 | 191 | 223 | 46.14% | 47.08% | 46.14% | 3.86 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 361 | 164 | 197 | 45.43% | 47.08% | 45.43% | 4.57 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 414 | 190 | 224 | 45.89% | 46.25% | 45.89% | 4.11 pp | -34 | 37 | -0.92 |
| BTC Daily | nn | NN | 589 | 277 | 312 | 47.03% | 45.42% | 48.12% | 2.97 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 414 | 189 | 225 | 45.65% | 47.92% | 45.65% | 4.35 pp | -36 | 37 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 766 | 361 | 405 | 47.13% | 43.33% | 47.50% | 2.87 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 766 | 358 | 408 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 361 | 156 | 205 | 43.21% | 43.33% | 43.21% | 6.79 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 361 | 154 | 207 | 42.66% | 41.67% | 42.66% | 7.34 pp | -53 | 37 | -1.43 |
| BTC Daily | lstm | LSTM | 589 | 264 | 325 | 44.82% | 44.58% | 44.79% | 5.18 pp | -61 | 37 | -1.65 |
| BTC Market Hours | xgb | XGBoost | 361 | 147 | 214 | 40.72% | 41.67% | 40.72% | 9.28 pp | -67 | 37 | -1.81 |
| BTC Hourly | rf | RandomForest | 766 | 343 | 423 | 44.78% | 44.58% | 44.58% | 5.22 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 766 | 342 | 424 | 44.65% | 40.42% | 45.42% | 5.35 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 414 | 170 | 244 | 41.06% | 40.42% | 41.06% | 8.94 pp | -74 | 37 | -2.00 |
| BTC Hourly | lstm | LSTM | 766 | 338 | 428 | 44.13% | 43.75% | 45.62% | 5.87 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 589 | 254 | 335 | 43.12% | 43.75% | 43.96% | 6.88 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 414 | 166 | 248 | 40.10% | 38.75% | 40.10% | 9.90 pp | -82 | 37 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 414 | 165 | 249 | 39.86% | 38.33% | 39.86% | 10.14 pp | -84 | 37 | -2.27 |
| BTC Hourly | xgb | XGBoost | 766 | 328 | 438 | 42.82% | 41.25% | 44.38% | 7.18 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 599 | 241 | 358 | 40.23% | 36.25% | 40.42% | 9.77 pp | -117 | 37 | -3.16 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 766 | 361 | 405 | 47.13% | 43.33% | 47.50% | 2.87 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 766 | 358 | 408 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 766 | 343 | 423 | 44.78% | 44.58% | 44.58% | 5.22 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 766 | 342 | 424 | 44.65% | 40.42% | 45.42% | 5.35 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 766 | 338 | 428 | 44.13% | 43.75% | 45.62% | 5.87 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 766 | 328 | 438 | 42.82% | 41.25% | 44.38% | 7.18 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 589 | 291 | 298 | 49.41% | 51.67% | 49.79% | 0.59 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 589 | 287 | 302 | 48.73% | 47.08% | 49.38% | 1.27 pp | -15 | 37 | -0.41 |
| BTC Daily | nn | NN | 589 | 277 | 312 | 47.03% | 45.42% | 48.12% | 2.97 pp | -35 | 37 | -0.95 |
| BTC Daily | lstm | LSTM | 589 | 264 | 325 | 44.82% | 44.58% | 44.79% | 5.18 pp | -61 | 37 | -1.65 |
| BTC Daily | rf | RandomForest | 589 | 254 | 335 | 43.12% | 43.75% | 43.96% | 6.88 pp | -81 | 37 | -2.19 |
| BTC Daily | xgb | XGBoost | 599 | 241 | 358 | 40.23% | 36.25% | 40.42% | 9.77 pp | -117 | 37 | -3.16 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 361 | 178 | 183 | 49.31% | 47.92% | 49.31% | 0.69 pp | -5 | 37 | -0.14 |
| BTC Market Hours | transformer | Transformer | 361 | 172 | 189 | 47.65% | 46.25% | 47.65% | 2.35 pp | -17 | 37 | -0.46 |
| BTC Market Hours | nn | NN | 361 | 164 | 197 | 45.43% | 47.08% | 45.43% | 4.57 pp | -33 | 37 | -0.89 |
| BTC Market Hours | lstm | LSTM | 361 | 156 | 205 | 43.21% | 43.33% | 43.21% | 6.79 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 361 | 154 | 207 | 42.66% | 41.67% | 42.66% | 7.34 pp | -53 | 37 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 361 | 147 | 214 | 40.72% | 41.67% | 40.72% | 9.28 pp | -67 | 37 | -1.81 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 414 | 191 | 223 | 46.14% | 47.08% | 46.14% | 3.86 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 414 | 190 | 224 | 45.89% | 46.25% | 45.89% | 4.11 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 414 | 189 | 225 | 45.65% | 47.92% | 45.65% | 4.35 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 414 | 170 | 244 | 41.06% | 40.42% | 41.06% | 8.94 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 414 | 166 | 248 | 40.10% | 38.75% | 40.10% | 9.90 pp | -82 | 37 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 414 | 165 | 249 | 39.86% | 38.33% | 39.86% | 10.14 pp | -84 | 37 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

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
