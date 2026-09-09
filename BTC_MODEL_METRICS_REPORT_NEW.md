# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T11:49:34.995697+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 291 | 231 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 327 | 267 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 476 | 255 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 476 | 255 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 223 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 21:00:00+00:00 | 223 | 76 | 147 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 255 | 134 | 121 | 52.55% | 52.92% | 52.55% | 2.55 pp | 13 | 20 | 0.65 |
| BTC Market Hours Daily | transformer | Transformer | 255 | 127 | 128 | 49.80% | 50.42% | 49.80% | 0.20 pp | -1 | 21 | -0.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 255 | 126 | 129 | 49.41% | 48.75% | 49.41% | 0.59 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | nn | NN | 255 | 124 | 131 | 48.63% | 48.33% | 48.63% | 1.37 pp | -7 | 21 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 255 | 121 | 134 | 47.45% | 47.92% | 47.45% | 2.55 pp | -13 | 20 | -0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 255 | 120 | 135 | 47.06% | 47.08% | 47.06% | 2.94 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 255 | 119 | 136 | 46.67% | 45.83% | 46.67% | 3.33 pp | -17 | 20 | -0.85 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 20 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 255 | 114 | 141 | 44.71% | 44.58% | 44.71% | 5.29 pp | -27 | 21 | -1.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 257 | 120 | 137 | 46.69% | 46.67% | 46.69% | 3.31 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 255 | 111 | 144 | 43.53% | 42.92% | 43.53% | 6.47 pp | -33 | 21 | -1.57 |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| BTC Daily | nn | NN | 257 | 117 | 140 | 45.53% | 44.17% | 45.53% | 4.47 pp | -23 | 12 | -1.92 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 255 | 104 | 151 | 40.78% | 41.25% | 40.78% | 9.22 pp | -47 | 21 | -2.24 |
| BTC Market Hours | lstm | LSTM | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 20 | -2.25 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| BTC Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 10 | -2.90 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | nn | NN | 231 | 97 | 134 | 41.99% | 41.99% | 41.99% | 8.01 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 257 | 104 | 153 | 40.47% | 40.00% | 40.47% | 9.53 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 257 | 97 | 160 | 37.74% | 36.67% | 37.74% | 12.26 pp | -63 | 12 | -5.25 |
| BTC Hourly | lstm | LSTM | 231 | 87 | 144 | 37.66% | 37.66% | 37.66% | 12.34 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 267 | 96 | 171 | 35.96% | 35.00% | 35.96% | 14.04 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 257 | 89 | 168 | 34.63% | 35.00% | 34.63% | 15.37 pp | -79 | 12 | -6.58 |
| BTC Hourly | xgb | XGBoost | 231 | 80 | 151 | 34.63% | 34.63% | 34.63% | 15.37 pp | -71 | 10 | -7.10 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 231 | 115 | 116 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 10 | -0.10 |
| BTC Hourly | transformer | Transformer | 231 | 101 | 130 | 43.72% | 43.72% | 43.72% | 6.28 pp | -29 | 10 | -2.90 |
| BTC Hourly | nn | NN | 231 | 97 | 134 | 41.99% | 41.99% | 41.99% | 8.01 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 231 | 96 | 135 | 41.56% | 41.56% | 41.56% | 8.44 pp | -39 | 10 | -3.90 |
| BTC Hourly | lstm | LSTM | 231 | 87 | 144 | 37.66% | 37.66% | 37.66% | 12.34 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 231 | 80 | 151 | 34.63% | 34.63% | 34.63% | 15.37 pp | -71 | 10 | -7.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 257 | 120 | 137 | 46.69% | 46.67% | 46.69% | 3.31 pp | -17 | 12 | -1.42 |
| BTC Daily | nn | NN | 257 | 117 | 140 | 45.53% | 44.17% | 45.53% | 4.47 pp | -23 | 12 | -1.92 |
| BTC Daily | transformer | Transformer | 257 | 104 | 153 | 40.47% | 40.00% | 40.47% | 9.53 pp | -49 | 12 | -4.08 |
| BTC Daily | rf | RandomForest | 257 | 97 | 160 | 37.74% | 36.67% | 37.74% | 12.26 pp | -63 | 12 | -5.25 |
| BTC Daily | xgb | XGBoost | 267 | 96 | 171 | 35.96% | 35.00% | 35.96% | 14.04 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 257 | 89 | 168 | 34.63% | 35.00% | 34.63% | 15.37 pp | -79 | 12 | -6.58 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 255 | 134 | 121 | 52.55% | 52.92% | 52.55% | 2.55 pp | 13 | 20 | 0.65 |
| BTC Market Hours | transformer | Transformer | 255 | 121 | 134 | 47.45% | 47.92% | 47.45% | 2.55 pp | -13 | 20 | -0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 255 | 120 | 135 | 47.06% | 47.08% | 47.06% | 2.94 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 255 | 119 | 136 | 46.67% | 45.83% | 46.67% | 3.33 pp | -17 | 20 | -0.85 |
| BTC Market Hours | rf | RandomForest | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 20 | -1.25 |
| BTC Market Hours | lstm | LSTM | 255 | 105 | 150 | 41.18% | 41.67% | 41.18% | 8.82 pp | -45 | 20 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 255 | 127 | 128 | 49.80% | 50.42% | 49.80% | 0.20 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 255 | 126 | 129 | 49.41% | 48.75% | 49.41% | 0.59 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | nn | NN | 255 | 124 | 131 | 48.63% | 48.33% | 48.63% | 1.37 pp | -7 | 21 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 255 | 114 | 141 | 44.71% | 44.58% | 44.71% | 5.29 pp | -27 | 21 | -1.29 |
| BTC Market Hours Daily | rf | RandomForest | 255 | 111 | 144 | 43.53% | 42.92% | 43.53% | 6.47 pp | -33 | 21 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 255 | 104 | 151 | 40.78% | 41.25% | 40.78% | 9.22 pp | -47 | 21 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 223 | 108 | 115 | 48.43% | 48.43% | 48.43% | 1.57 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 223 | 97 | 126 | 43.50% | 43.50% | 43.50% | 6.50 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 223 | 96 | 127 | 43.05% | 43.05% | 43.05% | 6.95 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 223 | 90 | 133 | 40.36% | 40.36% | 40.36% | 9.64 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 29 | 47 | 38.16% | 38.16% | 38.16% | 11.84 pp | -18 | 6 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
