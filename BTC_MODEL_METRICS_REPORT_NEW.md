# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T12:13:08.653222+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 307 | 247 | 60 | 0 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 270 | 132 | 138 | 48.89% | 48.33% | 48.89% | 1.11 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 270 | 131 | 139 | 48.52% | 48.33% | 48.52% | 1.48 pp | -8 | 22 | -0.36 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 247 | 120 | 127 | 48.58% | 48.75% | 48.58% | 1.42 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 237 | 113 | 124 | 47.68% | 47.68% | 47.68% | 2.32 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | transformer | Transformer | 270 | 126 | 144 | 46.67% | 45.42% | 46.67% | 3.33 pp | -18 | 22 | -0.82 |
| BTC Market Hours | transformer | Transformer | 270 | 126 | 144 | 46.67% | 46.67% | 46.67% | 3.33 pp | -18 | 21 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 270 | 125 | 145 | 46.30% | 46.25% | 46.30% | 3.70 pp | -20 | 21 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 270 | 124 | 146 | 45.93% | 45.00% | 45.93% | 4.07 pp | -22 | 21 | -1.05 |
| Consolidated Market Hours | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 84 | 38 | 46 | 45.24% | 45.24% | 45.24% | 4.76 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 237 | 109 | 128 | 45.99% | 45.99% | 45.99% | 4.01 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | rf | RandomForest | 270 | 121 | 149 | 44.81% | 43.33% | 44.81% | 5.19 pp | -28 | 22 | -1.27 |
| BTC Market Hours | rf | RandomForest | 270 | 119 | 151 | 44.07% | 42.50% | 44.07% | 5.93 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | xgb | XGBoost | 270 | 118 | 152 | 43.70% | 42.08% | 43.70% | 6.30 pp | -34 | 22 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 237 | 105 | 132 | 44.30% | 44.30% | 44.30% | 5.70 pp | -27 | 15 | -1.80 |
| BTC Daily | nn | NN | 272 | 125 | 147 | 45.96% | 45.42% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| BTC Daily | mlp_sklearn | MLPClassifier | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 7 | -2.00 |
| BTC Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 45.83% | 44.94% | 5.06 pp | -25 | 11 | -2.27 |
| BTC Market Hours | lstm | LSTM | 270 | 111 | 159 | 41.11% | 42.08% | 41.11% | 8.89 pp | -48 | 21 | -2.29 |
| Consolidated Market Hours | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 84 | 33 | 51 | 39.29% | 39.29% | 39.29% | 10.71 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 237 | 99 | 138 | 41.77% | 41.77% | 41.77% | 8.23 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 84 | 31 | 53 | 36.90% | 36.90% | 36.90% | 13.10 pp | -22 | 7 | -3.14 |
| Consolidated Hourly | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| Consolidated Daily/Hourly Refresh | nn | NN | 237 | 94 | 143 | 39.66% | 39.66% | 39.66% | 10.34 pp | -49 | 15 | -3.27 |
| BTC Market Hours Daily | lstm | LSTM | 270 | 99 | 171 | 36.67% | 37.92% | 36.67% | 13.33 pp | -72 | 22 | -3.27 |
| BTC Hourly | nn | NN | 247 | 105 | 142 | 42.51% | 42.50% | 42.51% | 7.49 pp | -37 | 11 | -3.36 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 84 | 30 | 54 | 35.71% | 35.71% | 35.71% | 14.29 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 7 | -3.71 |
| BTC Hourly | rf | RandomForest | 247 | 102 | 145 | 41.30% | 42.08% | 41.30% | 8.70 pp | -43 | 11 | -3.91 |
| BTC Daily | transformer | Transformer | 272 | 110 | 162 | 40.44% | 37.92% | 40.44% | 9.56 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 272 | 103 | 169 | 37.87% | 37.50% | 37.87% | 12.13 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 282 | 104 | 178 | 36.88% | 37.08% | 36.88% | 13.12 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 247 | 92 | 155 | 37.25% | 37.08% | 37.25% | 12.75 pp | -63 | 11 | -5.73 |
| BTC Hourly | xgb | XGBoost | 247 | 87 | 160 | 35.22% | 35.83% | 35.22% | 14.78 pp | -73 | 11 | -6.64 |
| BTC Daily | lstm | LSTM | 272 | 96 | 176 | 35.29% | 35.00% | 35.29% | 14.71 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 247 | 120 | 127 | 48.58% | 48.75% | 48.58% | 1.42 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 247 | 111 | 136 | 44.94% | 45.83% | 44.94% | 5.06 pp | -25 | 11 | -2.27 |
| BTC Hourly | nn | NN | 247 | 105 | 142 | 42.51% | 42.50% | 42.51% | 7.49 pp | -37 | 11 | -3.36 |
| BTC Hourly | rf | RandomForest | 247 | 102 | 145 | 41.30% | 42.08% | 41.30% | 8.70 pp | -43 | 11 | -3.91 |
| BTC Hourly | lstm | LSTM | 247 | 92 | 155 | 37.25% | 37.08% | 37.25% | 12.75 pp | -63 | 11 | -5.73 |
| BTC Hourly | xgb | XGBoost | 247 | 87 | 160 | 35.22% | 35.83% | 35.22% | 14.78 pp | -73 | 11 | -6.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 272 | 125 | 147 | 45.96% | 45.42% | 45.96% | 4.04 pp | -22 | 12 | -1.83 |
| BTC Daily | mlp_sklearn | MLPClassifier | 272 | 124 | 148 | 45.59% | 45.00% | 45.59% | 4.41 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 272 | 110 | 162 | 40.44% | 37.92% | 40.44% | 9.56 pp | -52 | 12 | -4.33 |
| BTC Daily | rf | RandomForest | 272 | 103 | 169 | 37.87% | 37.50% | 37.87% | 12.13 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 282 | 104 | 178 | 36.88% | 37.08% | 36.88% | 13.12 pp | -74 | 13 | -5.69 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 270 | 132 | 138 | 48.89% | 48.33% | 48.89% | 1.11 pp | -6 | 22 | -0.27 |
| BTC Market Hours Daily | nn | NN | 270 | 131 | 139 | 48.52% | 48.33% | 48.52% | 1.48 pp | -8 | 22 | -0.36 |
| BTC Market Hours Daily | transformer | Transformer | 270 | 126 | 144 | 46.67% | 45.42% | 46.67% | 3.33 pp | -18 | 22 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 270 | 121 | 149 | 44.81% | 43.33% | 44.81% | 5.19 pp | -28 | 22 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 270 | 118 | 152 | 43.70% | 42.08% | 43.70% | 6.30 pp | -34 | 22 | -1.55 |
| BTC Market Hours Daily | lstm | LSTM | 270 | 99 | 171 | 36.67% | 37.92% | 36.67% | 13.33 pp | -72 | 22 | -3.27 |

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
