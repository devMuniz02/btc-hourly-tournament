# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T23:18:55.924956+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 809 | 310 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 973 | 608 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 548 | 370 | 177 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 550 | 424 | 124 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 1 | 32 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 1 | 32 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 370 | 182 | 188 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Daily | transformer | Transformer | 598 | 295 | 303 | 49.33% | 50.42% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| Consolidated Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 598 | 292 | 306 | 48.83% | 47.08% | 49.38% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Market Hours | transformer | Transformer | 370 | 174 | 196 | 47.03% | 45.00% | 47.03% | 2.97 pp | -22 | 38 | -0.58 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 370 | 170 | 200 | 45.95% | 48.33% | 45.95% | 4.05 pp | -30 | 38 | -0.79 |
| BTC Daily | nn | NN | 598 | 282 | 316 | 47.16% | 45.42% | 48.33% | 2.84 pp | -34 | 38 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 424 | 193 | 231 | 45.52% | 45.00% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 424 | 193 | 231 | 45.52% | 46.25% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 424 | 193 | 231 | 45.52% | 47.50% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 775 | 365 | 410 | 47.10% | 43.75% | 47.08% | 2.90 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 775 | 363 | 412 | 46.84% | 42.92% | 45.83% | 3.16 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 370 | 158 | 212 | 42.70% | 42.92% | 42.70% | 7.30 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 370 | 157 | 213 | 42.43% | 40.83% | 42.43% | 7.57 pp | -56 | 38 | -1.47 |
| BTC Daily | lstm | LSTM | 598 | 267 | 331 | 44.65% | 43.75% | 44.79% | 5.35 pp | -64 | 38 | -1.68 |
| BTC Hourly | rf | RandomForest | 775 | 348 | 427 | 44.90% | 44.17% | 44.58% | 5.10 pp | -79 | 42 | -1.88 |
| BTC Market Hours | xgb | XGBoost | 370 | 149 | 221 | 40.27% | 40.83% | 40.27% | 9.73 pp | -72 | 38 | -1.89 |
| BTC Hourly | nn | NN | 775 | 347 | 428 | 44.77% | 39.58% | 45.62% | 5.23 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 598 | 259 | 339 | 43.31% | 44.58% | 43.96% | 6.69 pp | -80 | 38 | -2.11 |
| BTC Hourly | lstm | LSTM | 775 | 343 | 432 | 44.26% | 43.33% | 46.04% | 5.74 pp | -89 | 42 | -2.12 |
| BTC Market Hours Daily | rf | RandomForest | 424 | 171 | 253 | 40.33% | 39.58% | 40.33% | 9.67 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 424 | 168 | 256 | 39.62% | 38.33% | 39.62% | 10.38 pp | -88 | 38 | -2.32 |
| BTC Market Hours Daily | lstm | LSTM | 424 | 166 | 258 | 39.15% | 37.50% | 39.15% | 10.85 pp | -92 | 38 | -2.42 |
| BTC Hourly | xgb | XGBoost | 775 | 332 | 443 | 42.84% | 40.83% | 44.17% | 7.16 pp | -111 | 42 | -2.64 |
| Consolidated Hourly | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |
| BTC Daily | xgb | XGBoost | 608 | 245 | 363 | 40.30% | 36.25% | 40.21% | 9.70 pp | -118 | 38 | -3.11 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 775 | 365 | 410 | 47.10% | 43.75% | 47.08% | 2.90 pp | -45 | 42 | -1.07 |
| BTC Hourly | transformer | Transformer | 775 | 363 | 412 | 46.84% | 42.92% | 45.83% | 3.16 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 775 | 348 | 427 | 44.90% | 44.17% | 44.58% | 5.10 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 775 | 347 | 428 | 44.77% | 39.58% | 45.62% | 5.23 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 775 | 343 | 432 | 44.26% | 43.33% | 46.04% | 5.74 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 775 | 332 | 443 | 42.84% | 40.83% | 44.17% | 7.16 pp | -111 | 42 | -2.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 598 | 295 | 303 | 49.33% | 50.42% | 50.21% | 0.67 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 598 | 292 | 306 | 48.83% | 47.08% | 49.38% | 1.17 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 598 | 282 | 316 | 47.16% | 45.42% | 48.33% | 2.84 pp | -34 | 38 | -0.89 |
| BTC Daily | lstm | LSTM | 598 | 267 | 331 | 44.65% | 43.75% | 44.79% | 5.35 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 598 | 259 | 339 | 43.31% | 44.58% | 43.96% | 6.69 pp | -80 | 38 | -2.11 |
| BTC Daily | xgb | XGBoost | 608 | 245 | 363 | 40.30% | 36.25% | 40.21% | 9.70 pp | -118 | 38 | -3.11 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 370 | 182 | 188 | 49.19% | 48.75% | 49.19% | 0.81 pp | -6 | 38 | -0.16 |
| BTC Market Hours | transformer | Transformer | 370 | 174 | 196 | 47.03% | 45.00% | 47.03% | 2.97 pp | -22 | 38 | -0.58 |
| BTC Market Hours | nn | NN | 370 | 170 | 200 | 45.95% | 48.33% | 45.95% | 4.05 pp | -30 | 38 | -0.79 |
| BTC Market Hours | lstm | LSTM | 370 | 158 | 212 | 42.70% | 42.92% | 42.70% | 7.30 pp | -54 | 38 | -1.42 |
| BTC Market Hours | rf | RandomForest | 370 | 157 | 213 | 42.43% | 40.83% | 42.43% | 7.57 pp | -56 | 38 | -1.47 |
| BTC Market Hours | xgb | XGBoost | 370 | 149 | 221 | 40.27% | 40.83% | 40.27% | 9.73 pp | -72 | 38 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 424 | 193 | 231 | 45.52% | 45.00% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 424 | 193 | 231 | 45.52% | 46.25% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 424 | 193 | 231 | 45.52% | 47.50% | 45.52% | 4.48 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 424 | 171 | 253 | 40.33% | 39.58% | 40.33% | 9.67 pp | -82 | 38 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 424 | 168 | 256 | 39.62% | 38.33% | 39.62% | 10.38 pp | -88 | 38 | -2.32 |
| BTC Market Hours Daily | lstm | LSTM | 424 | 166 | 258 | 39.15% | 37.50% | 39.15% | 10.85 pp | -92 | 38 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
