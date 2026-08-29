# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T08:41:46.029068+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1123 | 835 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 999 | 634 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 589 | 396 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 591 | 450 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 55 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 1 | 54 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 16:00:00+00:00 | 55 | 1 | 54 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 396 | 195 | 201 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Daily | transformer | Transformer | 624 | 306 | 318 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 624 | 305 | 319 | 48.88% | 46.67% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 396 | 186 | 210 | 46.97% | 49.58% | 46.97% | 3.03 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 396 | 184 | 212 | 46.46% | 42.92% | 46.46% | 3.54 pp | -28 | 40 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 450 | 208 | 242 | 46.22% | 47.92% | 46.22% | 3.78 pp | -34 | 40 | -0.85 |
| BTC Daily | nn | NN | 624 | 294 | 330 | 47.12% | 43.75% | 49.17% | 2.88 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 450 | 206 | 244 | 45.78% | 45.00% | 45.78% | 4.22 pp | -38 | 40 | -0.95 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 450 | 204 | 246 | 45.33% | 45.42% | 45.33% | 4.67 pp | -42 | 40 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 801 | 377 | 424 | 47.07% | 44.58% | 46.88% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | transformer | Transformer | 801 | 377 | 424 | 47.07% | 44.58% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| Consolidated Hourly | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| BTC Market Hours | lstm | LSTM | 396 | 173 | 223 | 43.69% | 43.75% | 43.69% | 6.31 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 396 | 168 | 228 | 42.42% | 40.83% | 42.42% | 7.58 pp | -60 | 40 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 624 | 278 | 346 | 44.55% | 42.92% | 44.38% | 5.45 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 396 | 161 | 235 | 40.66% | 39.17% | 40.66% | 9.34 pp | -74 | 40 | -1.85 |
| BTC Hourly | nn | NN | 801 | 359 | 442 | 44.82% | 40.42% | 45.00% | 5.18 pp | -83 | 43 | -1.93 |
| BTC Hourly | rf | RandomForest | 801 | 356 | 445 | 44.44% | 43.33% | 44.17% | 5.56 pp | -89 | 43 | -2.07 |
| BTC Market Hours Daily | rf | RandomForest | 450 | 183 | 267 | 40.67% | 39.58% | 40.67% | 9.33 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 450 | 181 | 269 | 40.22% | 38.33% | 40.22% | 9.78 pp | -88 | 40 | -2.20 |
| BTC Hourly | lstm | LSTM | 801 | 352 | 449 | 43.95% | 43.75% | 45.21% | 6.05 pp | -97 | 43 | -2.26 |
| BTC Daily | rf | RandomForest | 624 | 268 | 356 | 42.95% | 42.92% | 43.75% | 7.05 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 450 | 177 | 273 | 39.33% | 37.50% | 39.33% | 10.67 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |
| BTC Hourly | xgb | XGBoost | 801 | 340 | 461 | 42.45% | 39.58% | 43.75% | 7.55 pp | -121 | 43 | -2.81 |
| BTC Daily | xgb | XGBoost | 634 | 250 | 384 | 39.43% | 32.50% | 40.00% | 10.57 pp | -134 | 39 | -3.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 801 | 377 | 424 | 47.07% | 44.58% | 46.88% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | transformer | Transformer | 801 | 377 | 424 | 47.07% | 44.58% | 46.46% | 2.93 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 801 | 359 | 442 | 44.82% | 40.42% | 45.00% | 5.18 pp | -83 | 43 | -1.93 |
| BTC Hourly | rf | RandomForest | 801 | 356 | 445 | 44.44% | 43.33% | 44.17% | 5.56 pp | -89 | 43 | -2.07 |
| BTC Hourly | lstm | LSTM | 801 | 352 | 449 | 43.95% | 43.75% | 45.21% | 6.05 pp | -97 | 43 | -2.26 |
| BTC Hourly | xgb | XGBoost | 801 | 340 | 461 | 42.45% | 39.58% | 43.75% | 7.55 pp | -121 | 43 | -2.81 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 624 | 306 | 318 | 49.04% | 47.50% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 624 | 305 | 319 | 48.88% | 46.67% | 50.21% | 1.12 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 624 | 294 | 330 | 47.12% | 43.75% | 49.17% | 2.88 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 624 | 278 | 346 | 44.55% | 42.92% | 44.38% | 5.45 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 624 | 268 | 356 | 42.95% | 42.92% | 43.75% | 7.05 pp | -88 | 39 | -2.26 |
| BTC Daily | xgb | XGBoost | 634 | 250 | 384 | 39.43% | 32.50% | 40.00% | 10.57 pp | -134 | 39 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 396 | 195 | 201 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 396 | 186 | 210 | 46.97% | 49.58% | 46.97% | 3.03 pp | -24 | 40 | -0.60 |
| BTC Market Hours | transformer | Transformer | 396 | 184 | 212 | 46.46% | 42.92% | 46.46% | 3.54 pp | -28 | 40 | -0.70 |
| BTC Market Hours | lstm | LSTM | 396 | 173 | 223 | 43.69% | 43.75% | 43.69% | 6.31 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 396 | 168 | 228 | 42.42% | 40.83% | 42.42% | 7.58 pp | -60 | 40 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 396 | 161 | 235 | 40.66% | 39.17% | 40.66% | 9.34 pp | -74 | 40 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 450 | 208 | 242 | 46.22% | 47.92% | 46.22% | 3.78 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 450 | 206 | 244 | 45.78% | 45.00% | 45.78% | 4.22 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 450 | 204 | 246 | 45.33% | 45.42% | 45.33% | 4.67 pp | -42 | 40 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 450 | 183 | 267 | 40.67% | 39.58% | 40.67% | 9.33 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 450 | 181 | 269 | 40.22% | 38.33% | 40.22% | 9.78 pp | -88 | 40 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 450 | 177 | 273 | 39.33% | 37.50% | 39.33% | 10.67 pp | -96 | 40 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 55 | 32 | 23 | 58.18% | 58.18% | 58.18% | 8.18 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 55 | 29 | 26 | 52.73% | 52.73% | 52.73% | 2.73 pp | 3 | 6 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 6 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 6 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
