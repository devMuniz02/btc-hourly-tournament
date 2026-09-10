# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T12:03:48.803003+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 306 | 246 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 342 | 282 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 504 | 270 | 234 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 00:00:00+00:00 | 504 | 270 | 234 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 237 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 16:00:00+00:00 | 237 | 84 | 153 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 270 | 141 | 129 | 52.22% | 51.67% | 52.22% | 2.22 pp | 12 | 21 | 0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 270 | 133 | 137 | 49.26% | 48.33% | 49.26% | 0.74 pp | -4 | 22 | -0.18 |
| BTC Market Hours Daily | transformer | Transformer | 270 | 131 | 139 | 48.52% | 47.92% | 48.52% | 1.48 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 270 | 130 | 140 | 48.15% | 48.33% | 48.15% | 1.85 pp | -10 | 22 | -0.45 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 11 | -0.55 |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| BTC Market Hours | transformer | Transformer | 270 | 126 | 144 | 46.67% | 46.67% | 46.67% | 3.33 pp | -18 | 21 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 270 | 125 | 145 | 46.30% | 46.25% | 46.30% | 3.70 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 270 | 124 | 146 | 45.93% | 45.00% | 45.93% | 4.07 pp | -22 | 21 | -1.05 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 270 | 119 | 151 | 44.07% | 42.92% | 44.07% | 5.93 pp | -32 | 22 | -1.45 |
| BTC Market Hours | rf | RandomForest | 270 | 119 | 151 | 44.07% | 42.50% | 44.07% | 5.93 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | rf | RandomForest | 270 | 116 | 154 | 42.96% | 41.67% | 42.96% | 7.04 pp | -38 | 22 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 272 | 125 | 147 | 45.96% | 45.00% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 272 | 125 | 147 | 45.96% | 45.42% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Hourly | transformer | Transformer | 246 | 111 | 135 | 45.12% | 45.83% | 45.12% | 4.88 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 270 | 110 | 160 | 40.74% | 42.08% | 40.74% | 9.26 pp | -50 | 22 | -2.27 |
| BTC Market Hours | lstm | LSTM | 270 | 111 | 159 | 41.11% | 42.08% | 41.11% | 8.89 pp | -48 | 21 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| BTC Hourly | nn | NN | 246 | 105 | 141 | 42.68% | 42.50% | 42.68% | 7.32 pp | -36 | 11 | -3.27 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Hourly | rf | RandomForest | 246 | 102 | 144 | 41.46% | 42.08% | 41.46% | 8.54 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 272 | 110 | 162 | 40.44% | 37.92% | 40.44% | 9.56 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 272 | 104 | 168 | 38.24% | 37.50% | 38.24% | 11.76 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 282 | 105 | 177 | 37.23% | 37.08% | 37.23% | 12.77 pp | -72 | 13 | -5.54 |
| BTC Hourly | lstm | LSTM | 246 | 92 | 154 | 37.40% | 37.08% | 37.40% | 12.60 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 246 | 87 | 159 | 35.37% | 35.83% | 35.37% | 14.63 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 272 | 96 | 176 | 35.29% | 35.00% | 35.29% | 14.71 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 246 | 120 | 126 | 48.78% | 48.75% | 48.78% | 1.22 pp | -6 | 11 | -0.55 |
| BTC Hourly | transformer | Transformer | 246 | 111 | 135 | 45.12% | 45.83% | 45.12% | 4.88 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 246 | 105 | 141 | 42.68% | 42.50% | 42.68% | 7.32 pp | -36 | 11 | -3.27 |
| BTC Hourly | rf | RandomForest | 246 | 102 | 144 | 41.46% | 42.08% | 41.46% | 8.54 pp | -42 | 11 | -3.82 |
| BTC Hourly | lstm | LSTM | 246 | 92 | 154 | 37.40% | 37.08% | 37.40% | 12.60 pp | -62 | 11 | -5.64 |
| BTC Hourly | xgb | XGBoost | 246 | 87 | 159 | 35.37% | 35.83% | 35.37% | 14.63 pp | -72 | 11 | -6.55 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 272 | 125 | 147 | 45.96% | 45.00% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| BTC Daily | nn | NN | 272 | 125 | 147 | 45.96% | 45.42% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| BTC Daily | transformer | Transformer | 272 | 110 | 162 | 40.44% | 37.92% | 40.44% | 9.56 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 272 | 104 | 168 | 38.24% | 37.50% | 38.24% | 11.76 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 282 | 105 | 177 | 37.23% | 37.08% | 37.23% | 12.77 pp | -72 | 13 | -5.54 |
| BTC Daily | lstm | LSTM | 272 | 96 | 176 | 35.29% | 35.00% | 35.29% | 14.71 pp | -80 | 12 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 270 | 141 | 129 | 52.22% | 51.67% | 52.22% | 2.22 pp | 12 | 21 | 0.57 |
| BTC Market Hours | transformer | Transformer | 270 | 126 | 144 | 46.67% | 46.67% | 46.67% | 3.33 pp | -18 | 21 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 270 | 125 | 145 | 46.30% | 46.25% | 46.30% | 3.70 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 270 | 124 | 146 | 45.93% | 45.00% | 45.93% | 4.07 pp | -22 | 21 | -1.05 |
| BTC Market Hours | rf | RandomForest | 270 | 119 | 151 | 44.07% | 42.50% | 44.07% | 5.93 pp | -32 | 21 | -1.52 |
| BTC Market Hours | lstm | LSTM | 270 | 111 | 159 | 41.11% | 42.08% | 41.11% | 8.89 pp | -48 | 21 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 270 | 133 | 137 | 49.26% | 48.33% | 49.26% | 0.74 pp | -4 | 22 | -0.18 |
| BTC Market Hours Daily | transformer | Transformer | 270 | 131 | 139 | 48.52% | 47.92% | 48.52% | 1.48 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | nn | NN | 270 | 130 | 140 | 48.15% | 48.33% | 48.15% | 1.85 pp | -10 | 22 | -0.45 |
| BTC Market Hours Daily | xgb | XGBoost | 270 | 119 | 151 | 44.07% | 42.92% | 44.07% | 5.93 pp | -32 | 22 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 270 | 116 | 154 | 42.96% | 41.67% | 42.96% | 7.04 pp | -38 | 22 | -1.73 |
| BTC Market Hours Daily | lstm | LSTM | 270 | 110 | 160 | 40.74% | 42.08% | 40.74% | 9.26 pp | -50 | 22 | -2.27 |

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
