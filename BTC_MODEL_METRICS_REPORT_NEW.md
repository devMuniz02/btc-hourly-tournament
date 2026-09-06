# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T22:39:35.267236+00:00
Scope: `new`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 250 | 190 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 285 | 225 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 21:00:00+00:00 | 405 | 213 | 192 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 21:00:00+00:00 | 405 | 213 | 192 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 183 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 13:00:00+00:00 | 183 | 55 | 128 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 213 | 110 | 103 | 51.64% | 51.64% | 51.64% | 1.64 pp | 7 | 18 | 0.39 |
| BTC Market Hours | nn | NN | 213 | 109 | 104 | 51.17% | 51.17% | 51.17% | 1.17 pp | 5 | 17 | 0.29 |
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 18 | -0.28 |
| BTC Market Hours | transformer | Transformer | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 17 | -0.29 |
| BTC Market Hours Daily | nn | NN | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 213 | 100 | 113 | 46.95% | 46.95% | 46.95% | 3.05 pp | -13 | 17 | -0.76 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| BTC Market Hours | rf | RandomForest | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 17 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 213 | 95 | 118 | 44.60% | 44.60% | 44.60% | 5.40 pp | -23 | 18 | -1.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 10 | -1.30 |
| BTC Market Hours | xgb | XGBoost | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 17 | -1.59 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| BTC Hourly | transformer | Transformer | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 8 | -2.00 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 213 | 87 | 126 | 40.85% | 40.85% | 40.85% | 9.15 pp | -39 | 18 | -2.17 |
| BTC Market Hours | lstm | LSTM | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 17 | -2.18 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 215 | 96 | 119 | 44.65% | 44.65% | 44.65% | 5.35 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | lstm | LSTM | 213 | 85 | 128 | 39.91% | 39.91% | 39.91% | 10.09 pp | -43 | 18 | -2.39 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 10 | -3.90 |
| BTC Hourly | rf | RandomForest | 190 | 79 | 111 | 41.58% | 41.58% | 41.58% | 8.42 pp | -32 | 8 | -4.00 |
| BTC Daily | rf | RandomForest | 215 | 82 | 133 | 38.14% | 38.14% | 38.14% | 11.86 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 225 | 80 | 145 | 35.56% | 35.56% | 35.56% | 14.44 pp | -65 | 11 | -5.91 |
| BTC Hourly | lstm | LSTM | 190 | 71 | 119 | 37.37% | 37.37% | 37.37% | 12.63 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 190 | 70 | 120 | 36.84% | 36.84% | 36.84% | 13.16 pp | -50 | 8 | -6.25 |
| BTC Daily | lstm | LSTM | 215 | 73 | 142 | 33.95% | 33.95% | 33.95% | 16.05 pp | -69 | 10 | -6.90 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 190 | 97 | 93 | 51.05% | 51.05% | 51.05% | 1.05 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 190 | 87 | 103 | 45.79% | 45.79% | 45.79% | 4.21 pp | -16 | 8 | -2.00 |
| BTC Hourly | nn | NN | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 190 | 79 | 111 | 41.58% | 41.58% | 41.58% | 8.42 pp | -32 | 8 | -4.00 |
| BTC Hourly | lstm | LSTM | 190 | 71 | 119 | 37.37% | 37.37% | 37.37% | 12.63 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 190 | 70 | 120 | 36.84% | 36.84% | 36.84% | 13.16 pp | -50 | 8 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 215 | 101 | 114 | 46.98% | 46.98% | 46.98% | 3.02 pp | -13 | 10 | -1.30 |
| BTC Daily | nn | NN | 215 | 96 | 119 | 44.65% | 44.65% | 44.65% | 5.35 pp | -23 | 10 | -2.30 |
| BTC Daily | transformer | Transformer | 215 | 88 | 127 | 40.93% | 40.93% | 40.93% | 9.07 pp | -39 | 10 | -3.90 |
| BTC Daily | rf | RandomForest | 215 | 82 | 133 | 38.14% | 38.14% | 38.14% | 11.86 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 225 | 80 | 145 | 35.56% | 35.56% | 35.56% | 14.44 pp | -65 | 11 | -5.91 |
| BTC Daily | lstm | LSTM | 215 | 73 | 142 | 33.95% | 33.95% | 33.95% | 16.05 pp | -69 | 10 | -6.90 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 213 | 109 | 104 | 51.17% | 51.17% | 51.17% | 1.17 pp | 5 | 17 | 0.29 |
| BTC Market Hours | transformer | Transformer | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 17 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 213 | 100 | 113 | 46.95% | 46.95% | 46.95% | 3.05 pp | -13 | 17 | -0.76 |
| BTC Market Hours | rf | RandomForest | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 17 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 17 | -1.59 |
| BTC Market Hours | lstm | LSTM | 213 | 88 | 125 | 41.31% | 41.31% | 41.31% | 8.69 pp | -37 | 17 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 213 | 110 | 103 | 51.64% | 51.64% | 51.64% | 1.64 pp | 7 | 18 | 0.39 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 213 | 104 | 109 | 48.83% | 48.83% | 48.83% | 1.17 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | nn | NN | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 213 | 95 | 118 | 44.60% | 44.60% | 44.60% | 5.40 pp | -23 | 18 | -1.28 |
| BTC Market Hours Daily | xgb | XGBoost | 213 | 87 | 126 | 40.85% | 40.85% | 40.85% | 9.15 pp | -39 | 18 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 213 | 85 | 128 | 39.91% | 39.91% | 39.91% | 10.09 pp | -43 | 18 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Hourly | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 183 | 91 | 92 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 13 | -0.85 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 55 | 25 | 30 | 45.45% | 45.45% | 45.45% | 4.55 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 20 | 35 | 36.36% | 36.36% | 36.36% | 13.64 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
