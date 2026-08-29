# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T05:45:14.300005+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1121 | 833 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 997 | 632 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 587 | 394 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 589 | 448 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 02:00:00+00:00 | 51 | 51 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 02:00:00+00:00 | 51 | 51 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 02:00:00+00:00 | 51 | 0 | 51 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 02:00:00+00:00 | 51 | 0 | 51 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 394 | 194 | 200 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| BTC Daily | transformer | Transformer | 622 | 306 | 316 | 49.20% | 48.33% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 622 | 304 | 318 | 48.87% | 47.08% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 394 | 184 | 210 | 46.70% | 48.75% | 46.70% | 3.30 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 394 | 183 | 211 | 46.45% | 43.33% | 46.45% | 3.55 pp | -28 | 40 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 6 | -0.83 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 6 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 448 | 207 | 241 | 46.21% | 48.33% | 46.21% | 3.79 pp | -34 | 40 | -0.85 |
| BTC Daily | nn | NN | 622 | 293 | 329 | 47.11% | 43.75% | 49.17% | 2.89 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 448 | 205 | 243 | 45.76% | 45.00% | 45.76% | 4.24 pp | -38 | 40 | -0.95 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 799 | 377 | 422 | 47.18% | 44.58% | 47.08% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 799 | 377 | 422 | 47.18% | 45.00% | 46.67% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | nn | NN | 448 | 202 | 246 | 45.09% | 45.00% | 45.09% | 4.91 pp | -44 | 40 | -1.10 |
| BTC Market Hours | lstm | LSTM | 394 | 172 | 222 | 43.65% | 43.75% | 43.65% | 6.35 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 394 | 167 | 227 | 42.39% | 40.83% | 42.39% | 7.61 pp | -60 | 40 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 622 | 277 | 345 | 44.53% | 43.33% | 44.38% | 5.47 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 40 | -1.85 |
| BTC Hourly | nn | NN | 799 | 359 | 440 | 44.93% | 40.83% | 45.21% | 5.07 pp | -81 | 43 | -1.88 |
| BTC Hourly | rf | RandomForest | 799 | 356 | 443 | 44.56% | 43.33% | 44.38% | 5.44 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 448 | 182 | 266 | 40.62% | 39.58% | 40.62% | 9.38 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 448 | 180 | 268 | 40.18% | 38.75% | 40.18% | 9.82 pp | -88 | 40 | -2.20 |
| BTC Hourly | lstm | LSTM | 799 | 352 | 447 | 44.06% | 44.17% | 45.62% | 5.94 pp | -95 | 43 | -2.21 |
| BTC Daily | rf | RandomForest | 622 | 267 | 355 | 42.93% | 42.92% | 43.75% | 7.07 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | xgb | XGBoost | 448 | 176 | 272 | 39.29% | 37.50% | 39.29% | 10.71 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |
| BTC Hourly | xgb | XGBoost | 799 | 340 | 459 | 42.55% | 39.58% | 44.17% | 7.45 pp | -119 | 43 | -2.77 |
| BTC Daily | xgb | XGBoost | 632 | 250 | 382 | 39.56% | 32.92% | 40.00% | 10.44 pp | -132 | 39 | -3.38 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 799 | 377 | 422 | 47.18% | 44.58% | 47.08% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 799 | 377 | 422 | 47.18% | 45.00% | 46.67% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | nn | NN | 799 | 359 | 440 | 44.93% | 40.83% | 45.21% | 5.07 pp | -81 | 43 | -1.88 |
| BTC Hourly | rf | RandomForest | 799 | 356 | 443 | 44.56% | 43.33% | 44.38% | 5.44 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 799 | 352 | 447 | 44.06% | 44.17% | 45.62% | 5.94 pp | -95 | 43 | -2.21 |
| BTC Hourly | xgb | XGBoost | 799 | 340 | 459 | 42.55% | 39.58% | 44.17% | 7.45 pp | -119 | 43 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 622 | 306 | 316 | 49.20% | 48.33% | 50.00% | 0.80 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 622 | 304 | 318 | 48.87% | 47.08% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 622 | 293 | 329 | 47.11% | 43.75% | 49.17% | 2.89 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 622 | 277 | 345 | 44.53% | 43.33% | 44.38% | 5.47 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 622 | 267 | 355 | 42.93% | 42.92% | 43.75% | 7.07 pp | -88 | 39 | -2.26 |
| BTC Daily | xgb | XGBoost | 632 | 250 | 382 | 39.56% | 32.92% | 40.00% | 10.44 pp | -132 | 39 | -3.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 394 | 194 | 200 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 394 | 184 | 210 | 46.70% | 48.75% | 46.70% | 3.30 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 394 | 183 | 211 | 46.45% | 43.33% | 46.45% | 3.55 pp | -28 | 40 | -0.70 |
| BTC Market Hours | lstm | LSTM | 394 | 172 | 222 | 43.65% | 43.75% | 43.65% | 6.35 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 394 | 167 | 227 | 42.39% | 40.83% | 42.39% | 7.61 pp | -60 | 40 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 40 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 448 | 207 | 241 | 46.21% | 48.33% | 46.21% | 3.79 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 448 | 205 | 243 | 45.76% | 45.00% | 45.76% | 4.24 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 448 | 202 | 246 | 45.09% | 45.00% | 45.09% | 4.91 pp | -44 | 40 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 448 | 182 | 266 | 40.62% | 39.58% | 40.62% | 9.38 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 448 | 180 | 268 | 40.18% | 38.75% | 40.18% | 9.82 pp | -88 | 40 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 448 | 176 | 272 | 39.29% | 37.50% | 39.29% | 10.71 pp | -96 | 40 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Hourly | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 6 | -0.83 |
| Consolidated Hourly | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 51 | 28 | 23 | 54.90% | 54.90% | 54.90% | 4.90 pp | 5 | 6 | 0.83 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 51 | 26 | 25 | 50.98% | 50.98% | 50.98% | 0.98 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 51 | 25 | 26 | 49.02% | 49.02% | 49.02% | 0.98 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 51 | 23 | 28 | 45.10% | 45.10% | 45.10% | 4.90 pp | -5 | 6 | -0.83 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 51 | 18 | 33 | 35.29% | 35.29% | 35.29% | 14.71 pp | -15 | 6 | -2.50 |

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
