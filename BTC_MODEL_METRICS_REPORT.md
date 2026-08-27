# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T10:39:50.735061+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 530 | 415 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T09:00:00+00:00 | 25 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T09:00:00+00:00 | 25 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T09:00:00+00:00 | 25 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T09:00:00+00:00 | 26 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 361 | 178 | 183 | 49.31% | 47.92% | 49.31% | 0.69 pp | -5 | 37 | -0.14 |
| BTC Daily | transformer | Transformer | 589 | 291 | 298 | 49.41% | 51.67% | 49.79% | 0.59 pp | -7 | 37 | -0.19 |
| BTC Daily | mlp_sklearn | MLPClassifier | 589 | 287 | 302 | 48.73% | 47.08% | 49.38% | 1.27 pp | -15 | 37 | -0.41 |
| BTC Market Hours | transformer | Transformer | 361 | 172 | 189 | 47.65% | 46.25% | 47.65% | 2.35 pp | -17 | 37 | -0.46 |
| BTC Market Hours Daily | nn | NN | 415 | 191 | 224 | 46.02% | 47.08% | 46.02% | 3.98 pp | -33 | 37 | -0.89 |
| BTC Market Hours | nn | NN | 361 | 164 | 197 | 45.43% | 47.08% | 45.43% | 4.57 pp | -33 | 37 | -0.89 |
| BTC Daily | nn | NN | 589 | 277 | 312 | 47.03% | 45.42% | 48.12% | 2.97 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 415 | 190 | 225 | 45.78% | 45.83% | 45.78% | 4.22 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 415 | 190 | 225 | 45.78% | 47.92% | 45.78% | 4.22 pp | -35 | 37 | -0.95 |
| Consolidated Hourly | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 766 | 361 | 405 | 47.13% | 43.33% | 47.50% | 2.87 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 766 | 358 | 408 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 361 | 156 | 205 | 43.21% | 43.33% | 43.21% | 6.79 pp | -49 | 37 | -1.32 |
| BTC Market Hours | rf | RandomForest | 361 | 154 | 207 | 42.66% | 41.67% | 42.66% | 7.34 pp | -53 | 37 | -1.43 |
| BTC Daily | lstm | LSTM | 589 | 264 | 325 | 44.82% | 44.58% | 44.79% | 5.18 pp | -61 | 37 | -1.65 |
| BTC Market Hours | xgb | XGBoost | 361 | 147 | 214 | 40.72% | 41.67% | 40.72% | 9.28 pp | -67 | 37 | -1.81 |
| BTC Hourly | rf | RandomForest | 766 | 343 | 423 | 44.78% | 44.58% | 44.58% | 5.22 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 766 | 342 | 424 | 44.65% | 40.42% | 45.42% | 5.35 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 415 | 170 | 245 | 40.96% | 40.42% | 40.96% | 9.04 pp | -75 | 37 | -2.03 |
| BTC Hourly | lstm | LSTM | 766 | 338 | 428 | 44.13% | 43.75% | 45.62% | 5.87 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 589 | 254 | 335 | 43.12% | 43.75% | 43.96% | 6.88 pp | -81 | 37 | -2.19 |
| BTC Market Hours Daily | lstm | LSTM | 415 | 166 | 249 | 40.00% | 38.33% | 40.00% | 10.00 pp | -83 | 37 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 415 | 166 | 249 | 40.00% | 38.75% | 40.00% | 10.00 pp | -83 | 37 | -2.24 |
| BTC Hourly | xgb | XGBoost | 766 | 328 | 438 | 42.82% | 41.25% | 44.38% | 7.18 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 599 | 241 | 358 | 40.23% | 36.25% | 40.42% | 9.77 pp | -117 | 37 | -3.16 |
| Consolidated Hourly | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |

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
| BTC Market Hours Daily | nn | NN | 415 | 191 | 224 | 46.02% | 47.08% | 46.02% | 3.98 pp | -33 | 37 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 415 | 190 | 225 | 45.78% | 45.83% | 45.78% | 4.22 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | transformer | Transformer | 415 | 190 | 225 | 45.78% | 47.92% | 45.78% | 4.22 pp | -35 | 37 | -0.95 |
| BTC Market Hours Daily | rf | RandomForest | 415 | 170 | 245 | 40.96% | 40.42% | 40.96% | 9.04 pp | -75 | 37 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 415 | 166 | 249 | 40.00% | 38.33% | 40.00% | 10.00 pp | -83 | 37 | -2.24 |
| BTC Market Hours Daily | xgb | XGBoost | 415 | 166 | 249 | 40.00% | 38.75% | 40.00% | 10.00 pp | -83 | 37 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 25 | 16 | 9 | 64.00% | 64.00% | 64.00% | 14.00 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 25 | 13 | 12 | 52.00% | 52.00% | 52.00% | 2.00 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 25 | 11 | 14 | 44.00% | 44.00% | 44.00% | 6.00 pp | -3 | 3 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 25 | 7 | 18 | 28.00% | 28.00% | 28.00% | 22.00 pp | -11 | 3 | -3.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
