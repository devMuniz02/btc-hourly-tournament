# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T08:12:55.213592+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 320 | 260 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 355 | 295 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 530 | 283 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 530 | 283 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 249 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 22:00:00+00:00 | 249 | 90 | 159 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 283 | 145 | 138 | 51.24% | 50.42% | 51.24% | 1.24 pp | 7 | 22 | 0.32 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 283 | 140 | 143 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 283 | 137 | 146 | 48.41% | 49.58% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 283 | 137 | 146 | 48.41% | 48.33% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 283 | 132 | 151 | 46.64% | 47.08% | 46.64% | 3.36 pp | -19 | 22 | -0.86 |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 260 | 125 | 135 | 48.08% | 47.92% | 48.08% | 1.92 pp | -10 | 11 | -0.91 |
| BTC Market Hours | transformer | Transformer | 283 | 131 | 152 | 46.29% | 46.25% | 46.29% | 3.71 pp | -21 | 22 | -0.95 |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| BTC Market Hours | xgb | XGBoost | 283 | 128 | 155 | 45.23% | 45.83% | 45.23% | 4.77 pp | -27 | 22 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| BTC Market Hours | rf | RandomForest | 283 | 126 | 157 | 44.52% | 43.75% | 44.52% | 5.48 pp | -31 | 22 | -1.41 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| BTC Market Hours Daily | rf | RandomForest | 283 | 122 | 161 | 43.11% | 43.33% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 283 | 122 | 161 | 43.11% | 42.92% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Daily | nn | NN | 285 | 130 | 155 | 45.61% | 45.00% | 45.61% | 4.39 pp | -25 | 13 | -1.92 |
| BTC Hourly | transformer | Transformer | 260 | 119 | 141 | 45.77% | 46.25% | 45.77% | 4.23 pp | -22 | 11 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 283 | 117 | 166 | 41.34% | 44.17% | 41.34% | 8.66 pp | -49 | 23 | -2.13 |
| BTC Market Hours | lstm | LSTM | 283 | 117 | 166 | 41.34% | 43.33% | 41.34% | 8.66 pp | -49 | 22 | -2.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 285 | 128 | 157 | 44.91% | 44.58% | 44.91% | 5.09 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |
| BTC Hourly | nn | NN | 260 | 109 | 151 | 41.92% | 41.67% | 41.92% | 8.08 pp | -42 | 11 | -3.82 |
| BTC Daily | transformer | Transformer | 285 | 114 | 171 | 40.00% | 37.92% | 40.00% | 10.00 pp | -57 | 13 | -4.38 |
| BTC Hourly | rf | RandomForest | 260 | 105 | 155 | 40.38% | 40.42% | 40.38% | 9.62 pp | -50 | 11 | -4.55 |
| BTC Daily | rf | RandomForest | 285 | 108 | 177 | 37.89% | 37.08% | 37.89% | 12.11 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 295 | 110 | 185 | 37.29% | 37.50% | 37.29% | 12.71 pp | -75 | 14 | -5.36 |
| BTC Hourly | lstm | LSTM | 260 | 95 | 165 | 36.54% | 35.42% | 36.54% | 13.46 pp | -70 | 11 | -6.36 |
| BTC Daily | lstm | LSTM | 285 | 101 | 184 | 35.44% | 35.83% | 35.44% | 14.56 pp | -83 | 13 | -6.38 |
| BTC Hourly | xgb | XGBoost | 260 | 89 | 171 | 34.23% | 34.58% | 34.23% | 15.77 pp | -82 | 11 | -7.45 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 260 | 125 | 135 | 48.08% | 47.92% | 48.08% | 1.92 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 260 | 119 | 141 | 45.77% | 46.25% | 45.77% | 4.23 pp | -22 | 11 | -2.00 |
| BTC Hourly | nn | NN | 260 | 109 | 151 | 41.92% | 41.67% | 41.92% | 8.08 pp | -42 | 11 | -3.82 |
| BTC Hourly | rf | RandomForest | 260 | 105 | 155 | 40.38% | 40.42% | 40.38% | 9.62 pp | -50 | 11 | -4.55 |
| BTC Hourly | lstm | LSTM | 260 | 95 | 165 | 36.54% | 35.42% | 36.54% | 13.46 pp | -70 | 11 | -6.36 |
| BTC Hourly | xgb | XGBoost | 260 | 89 | 171 | 34.23% | 34.58% | 34.23% | 15.77 pp | -82 | 11 | -7.45 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 285 | 130 | 155 | 45.61% | 45.00% | 45.61% | 4.39 pp | -25 | 13 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 285 | 128 | 157 | 44.91% | 44.58% | 44.91% | 5.09 pp | -29 | 13 | -2.23 |
| BTC Daily | transformer | Transformer | 285 | 114 | 171 | 40.00% | 37.92% | 40.00% | 10.00 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 285 | 108 | 177 | 37.89% | 37.08% | 37.89% | 12.11 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 295 | 110 | 185 | 37.29% | 37.50% | 37.29% | 12.71 pp | -75 | 14 | -5.36 |
| BTC Daily | lstm | LSTM | 285 | 101 | 184 | 35.44% | 35.83% | 35.44% | 14.56 pp | -83 | 13 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 283 | 145 | 138 | 51.24% | 50.42% | 51.24% | 1.24 pp | 7 | 22 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 283 | 132 | 151 | 46.64% | 47.08% | 46.64% | 3.36 pp | -19 | 22 | -0.86 |
| BTC Market Hours | transformer | Transformer | 283 | 131 | 152 | 46.29% | 46.25% | 46.29% | 3.71 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 283 | 128 | 155 | 45.23% | 45.83% | 45.23% | 4.77 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 283 | 126 | 157 | 44.52% | 43.75% | 44.52% | 5.48 pp | -31 | 22 | -1.41 |
| BTC Market Hours | lstm | LSTM | 283 | 117 | 166 | 41.34% | 43.33% | 41.34% | 8.66 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 283 | 140 | 143 | 49.47% | 49.17% | 49.47% | 0.53 pp | -3 | 23 | -0.13 |
| BTC Market Hours Daily | nn | NN | 283 | 137 | 146 | 48.41% | 49.58% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | transformer | Transformer | 283 | 137 | 146 | 48.41% | 48.33% | 48.41% | 1.59 pp | -9 | 23 | -0.39 |
| BTC Market Hours Daily | rf | RandomForest | 283 | 122 | 161 | 43.11% | 43.33% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | xgb | XGBoost | 283 | 122 | 161 | 43.11% | 42.92% | 43.11% | 6.89 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 283 | 117 | 166 | 41.34% | 44.17% | 41.34% | 8.66 pp | -49 | 23 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Hourly | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 249 | 118 | 131 | 47.39% | 47.50% | 47.39% | 2.61 pp | -13 | 15 | -0.87 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 249 | 114 | 135 | 45.78% | 45.00% | 45.78% | 4.22 pp | -21 | 15 | -1.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 249 | 113 | 136 | 45.38% | 45.83% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 249 | 113 | 136 | 45.38% | 45.00% | 45.38% | 4.62 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 249 | 103 | 146 | 41.37% | 41.25% | 41.37% | 8.63 pp | -43 | 15 | -2.87 |
| Consolidated Daily/Hourly Refresh | nn | NN | 249 | 101 | 148 | 40.56% | 41.25% | 40.56% | 9.44 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 7 | -1.14 |
| Consolidated Market Hours Daily | rf | RandomForest | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 7 | -3.71 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
