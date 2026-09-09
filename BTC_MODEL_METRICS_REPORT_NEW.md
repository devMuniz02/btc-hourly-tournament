# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T10:35:09.764874+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 290 | 230 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 326 | 266 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 475 | 254 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 475 | 254 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 222 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T20:00:00+00:00 | 223 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 254 | 133 | 121 | 52.36% | 52.50% | 52.36% | 2.36 pp | 12 | 20 | 0.60 |
| BTC Market Hours Daily | transformer | Transformer | 254 | 127 | 127 | 50.00% | 50.42% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 230 | 115 | 115 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 254 | 125 | 129 | 49.21% | 48.33% | 49.21% | 0.79 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 254 | 123 | 131 | 48.43% | 47.92% | 48.43% | 1.57 pp | -8 | 21 | -0.38 |
| BTC Market Hours | transformer | Transformer | 254 | 121 | 133 | 47.64% | 48.33% | 47.64% | 2.36 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 254 | 119 | 135 | 46.85% | 46.67% | 46.85% | 3.15 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 254 | 119 | 135 | 46.85% | 45.83% | 46.85% | 3.15 pp | -16 | 20 | -0.80 |
| Consolidated Hourly | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| BTC Market Hours | rf | RandomForest | 254 | 114 | 140 | 44.88% | 44.17% | 44.88% | 5.12 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 254 | 113 | 141 | 44.49% | 44.58% | 44.49% | 5.51 pp | -28 | 21 | -1.33 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| BTC Market Hours Daily | rf | RandomForest | 254 | 110 | 144 | 43.31% | 42.50% | 43.31% | 6.69 pp | -34 | 21 | -1.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 11 | -1.64 |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Daily | nn | NN | 256 | 116 | 140 | 45.31% | 43.75% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 254 | 105 | 149 | 41.34% | 41.67% | 41.34% | 8.66 pp | -44 | 20 | -2.20 |
| Consolidated Hourly | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| BTC Market Hours Daily | lstm | LSTM | 254 | 103 | 151 | 40.55% | 40.83% | 40.55% | 9.45 pp | -48 | 21 | -2.29 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
| BTC Hourly | transformer | Transformer | 230 | 101 | 129 | 43.91% | 43.91% | 43.91% | 6.09 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 76 | 28 | 48 | 36.84% | 36.84% | 36.84% | 13.16 pp | -20 | 6 | -3.33 |
| BTC Hourly | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 230 | 95 | 135 | 41.30% | 41.30% | 41.30% | 8.70 pp | -40 | 10 | -4.00 |
| BTC Daily | transformer | Transformer | 256 | 103 | 153 | 40.23% | 39.58% | 40.23% | 9.77 pp | -50 | 11 | -4.55 |
| BTC Hourly | lstm | LSTM | 230 | 86 | 144 | 37.39% | 37.39% | 37.39% | 12.61 pp | -58 | 10 | -5.80 |
| BTC Daily | rf | RandomForest | 256 | 96 | 160 | 37.50% | 36.25% | 37.50% | 12.50 pp | -64 | 11 | -5.82 |
| BTC Daily | xgb | XGBoost | 266 | 95 | 171 | 35.71% | 35.00% | 35.71% | 14.29 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 230 | 80 | 150 | 34.78% | 34.78% | 34.78% | 15.22 pp | -70 | 10 | -7.00 |
| BTC Daily | lstm | LSTM | 256 | 88 | 168 | 34.38% | 34.58% | 34.38% | 15.62 pp | -80 | 11 | -7.27 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 230 | 115 | 115 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Hourly | transformer | Transformer | 230 | 101 | 129 | 43.91% | 43.91% | 43.91% | 6.09 pp | -28 | 10 | -2.80 |
| BTC Hourly | nn | NN | 230 | 96 | 134 | 41.74% | 41.74% | 41.74% | 8.26 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 230 | 95 | 135 | 41.30% | 41.30% | 41.30% | 8.70 pp | -40 | 10 | -4.00 |
| BTC Hourly | lstm | LSTM | 230 | 86 | 144 | 37.39% | 37.39% | 37.39% | 12.61 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 230 | 80 | 150 | 34.78% | 34.78% | 34.78% | 15.22 pp | -70 | 10 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 11 | -1.64 |
| BTC Daily | nn | NN | 256 | 116 | 140 | 45.31% | 43.75% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 256 | 103 | 153 | 40.23% | 39.58% | 40.23% | 9.77 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 256 | 96 | 160 | 37.50% | 36.25% | 37.50% | 12.50 pp | -64 | 11 | -5.82 |
| BTC Daily | xgb | XGBoost | 266 | 95 | 171 | 35.71% | 35.00% | 35.71% | 14.29 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 256 | 88 | 168 | 34.38% | 34.58% | 34.38% | 15.62 pp | -80 | 11 | -7.27 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 254 | 133 | 121 | 52.36% | 52.50% | 52.36% | 2.36 pp | 12 | 20 | 0.60 |
| BTC Market Hours | transformer | Transformer | 254 | 121 | 133 | 47.64% | 48.33% | 47.64% | 2.36 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 254 | 119 | 135 | 46.85% | 46.67% | 46.85% | 3.15 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 254 | 119 | 135 | 46.85% | 45.83% | 46.85% | 3.15 pp | -16 | 20 | -0.80 |
| BTC Market Hours | rf | RandomForest | 254 | 114 | 140 | 44.88% | 44.17% | 44.88% | 5.12 pp | -26 | 20 | -1.30 |
| BTC Market Hours | lstm | LSTM | 254 | 105 | 149 | 41.34% | 41.67% | 41.34% | 8.66 pp | -44 | 20 | -2.20 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 254 | 127 | 127 | 50.00% | 50.42% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 254 | 125 | 129 | 49.21% | 48.33% | 49.21% | 0.79 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 254 | 123 | 131 | 48.43% | 47.92% | 48.43% | 1.57 pp | -8 | 21 | -0.38 |
| BTC Market Hours Daily | xgb | XGBoost | 254 | 113 | 141 | 44.49% | 44.58% | 44.49% | 5.51 pp | -28 | 21 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 254 | 110 | 144 | 43.31% | 42.50% | 43.31% | 6.69 pp | -34 | 21 | -1.62 |
| BTC Market Hours Daily | lstm | LSTM | 254 | 103 | 151 | 40.55% | 40.83% | 40.55% | 9.45 pp | -48 | 21 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 222 | 105 | 117 | 47.30% | 47.30% | 47.30% | 2.70 pp | -12 | 14 | -0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 14 | -1.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 222 | 95 | 127 | 42.79% | 42.79% | 42.79% | 7.21 pp | -32 | 14 | -2.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 222 | 94 | 128 | 42.34% | 42.34% | 42.34% | 7.66 pp | -34 | 14 | -2.43 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 76 | 32 | 44 | 42.11% | 42.11% | 42.11% | 7.89 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 6 | -2.67 |
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
