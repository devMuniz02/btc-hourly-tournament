# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T13:54:14.735741+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 308 | 248 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 344 | 284 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 507 | 272 | 235 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 12:00:00+00:00 | 507 | 272 | 235 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 272 | 141 | 131 | 51.84% | 51.67% | 51.84% | 1.84 pp | 10 | 21 | 0.48 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 272 | 133 | 139 | 48.90% | 47.92% | 48.90% | 1.10 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 272 | 132 | 140 | 48.53% | 47.92% | 48.53% | 1.47 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 272 | 130 | 142 | 47.79% | 48.33% | 47.79% | 2.21 pp | -12 | 22 | -0.55 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 248 | 120 | 128 | 48.39% | 48.33% | 48.39% | 1.61 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 272 | 126 | 146 | 46.32% | 46.25% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | transformer | Transformer | 272 | 126 | 146 | 46.32% | 45.83% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 21 | -1.14 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 272 | 119 | 153 | 43.75% | 42.50% | 43.75% | 6.25 pp | -34 | 22 | -1.55 |
| BTC Market Hours | rf | RandomForest | 272 | 119 | 153 | 43.75% | 42.08% | 43.75% | 6.25 pp | -34 | 21 | -1.62 |
| BTC Daily | nn | NN | 274 | 127 | 147 | 46.35% | 45.42% | 46.35% | 3.65 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 272 | 116 | 156 | 42.65% | 41.67% | 42.65% | 7.35 pp | -40 | 22 | -1.82 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 125 | 149 | 45.62% | 45.00% | 45.62% | 4.38 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Hourly | transformer | Transformer | 248 | 112 | 136 | 45.16% | 46.25% | 45.16% | 4.84 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 272 | 112 | 160 | 41.18% | 42.50% | 41.18% | 8.82 pp | -48 | 22 | -2.18 |
| BTC Market Hours | lstm | LSTM | 272 | 113 | 159 | 41.54% | 42.50% | 41.54% | 8.46 pp | -46 | 21 | -2.19 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 248 | 105 | 143 | 42.34% | 42.50% | 42.34% | 7.66 pp | -38 | 11 | -3.45 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Hourly | rf | RandomForest | 248 | 103 | 145 | 41.53% | 42.50% | 41.53% | 8.47 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 274 | 111 | 163 | 40.51% | 37.92% | 40.51% | 9.49 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 274 | 104 | 170 | 37.96% | 37.50% | 37.96% | 12.04 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 284 | 105 | 179 | 36.97% | 36.67% | 36.97% | 13.03 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 248 | 92 | 156 | 37.10% | 36.67% | 37.10% | 12.90 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 248 | 88 | 160 | 35.48% | 36.25% | 35.48% | 14.52 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 274 | 96 | 178 | 35.04% | 35.00% | 35.04% | 14.96 pp | -82 | 12 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 248 | 120 | 128 | 48.39% | 48.33% | 48.39% | 1.61 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 248 | 112 | 136 | 45.16% | 46.25% | 45.16% | 4.84 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 248 | 105 | 143 | 42.34% | 42.50% | 42.34% | 7.66 pp | -38 | 11 | -3.45 |
| BTC Hourly | rf | RandomForest | 248 | 103 | 145 | 41.53% | 42.50% | 41.53% | 8.47 pp | -42 | 11 | -3.82 |
| BTC Hourly | lstm | LSTM | 248 | 92 | 156 | 37.10% | 36.67% | 37.10% | 12.90 pp | -64 | 11 | -5.82 |
| BTC Hourly | xgb | XGBoost | 248 | 88 | 160 | 35.48% | 36.25% | 35.48% | 14.52 pp | -72 | 11 | -6.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 274 | 127 | 147 | 46.35% | 45.42% | 46.35% | 3.65 pp | -20 | 12 | -1.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 274 | 125 | 149 | 45.62% | 45.00% | 45.62% | 4.38 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 274 | 111 | 163 | 40.51% | 37.92% | 40.51% | 9.49 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 274 | 104 | 170 | 37.96% | 37.50% | 37.96% | 12.04 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 284 | 105 | 179 | 36.97% | 36.67% | 36.97% | 13.03 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 274 | 96 | 178 | 35.04% | 35.00% | 35.04% | 14.96 pp | -82 | 12 | -6.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 272 | 141 | 131 | 51.84% | 51.67% | 51.84% | 1.84 pp | 10 | 21 | 0.48 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 272 | 126 | 146 | 46.32% | 46.25% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | transformer | Transformer | 272 | 126 | 146 | 46.32% | 45.83% | 46.32% | 3.68 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 21 | -1.14 |
| BTC Market Hours | rf | RandomForest | 272 | 119 | 153 | 43.75% | 42.08% | 43.75% | 6.25 pp | -34 | 21 | -1.62 |
| BTC Market Hours | lstm | LSTM | 272 | 113 | 159 | 41.54% | 42.50% | 41.54% | 8.46 pp | -46 | 21 | -2.19 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 272 | 133 | 139 | 48.90% | 47.92% | 48.90% | 1.10 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 272 | 132 | 140 | 48.53% | 47.92% | 48.53% | 1.47 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 272 | 130 | 142 | 47.79% | 48.33% | 47.79% | 2.21 pp | -12 | 22 | -0.55 |
| BTC Market Hours Daily | xgb | XGBoost | 272 | 119 | 153 | 43.75% | 42.50% | 43.75% | 6.25 pp | -34 | 22 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 272 | 116 | 156 | 42.65% | 41.67% | 42.65% | 7.35 pp | -40 | 22 | -1.82 |
| BTC Market Hours Daily | lstm | LSTM | 272 | 112 | 160 | 41.18% | 42.50% | 41.18% | 8.82 pp | -48 | 22 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
