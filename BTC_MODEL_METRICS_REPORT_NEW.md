# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T02:32:49.094449+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 219 | 159 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 255 | 195 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 352 | 183 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 352 | 183 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 159 | 83 | 76 | 52.20% | 52.20% | 52.20% | 2.20 pp | 7 | 7 | 1.00 |
| BTC Market Hours Daily | transformer | Transformer | 183 | 95 | 88 | 51.91% | 51.91% | 51.91% | 1.91 pp | 7 | 15 | 0.47 |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| BTC Market Hours | nn | NN | 183 | 93 | 90 | 50.82% | 50.82% | 50.82% | 0.82 pp | 3 | 15 | 0.20 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | transformer | Transformer | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 15 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 15 | -0.33 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 15 | -1.00 |
| BTC Market Hours | rf | RandomForest | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 15 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 15 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 9 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 15 | -1.80 |
| BTC Market Hours | lstm | LSTM | 183 | 77 | 106 | 42.08% | 42.08% | 42.08% | 7.92 pp | -29 | 15 | -1.93 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 183 | 76 | 107 | 41.53% | 41.53% | 41.53% | 8.47 pp | -31 | 15 | -2.07 |
| BTC Daily | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 9 | -2.11 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 183 | 74 | 109 | 40.44% | 40.44% | 40.44% | 9.56 pp | -35 | 15 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 9 | -2.78 |
| BTC Hourly | nn | NN | 159 | 68 | 91 | 42.77% | 42.77% | 42.77% | 7.23 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 185 | 73 | 112 | 39.46% | 39.46% | 39.46% | 10.54 pp | -39 | 9 | -4.33 |
| BTC Daily | xgb | XGBoost | 195 | 71 | 124 | 36.41% | 36.41% | 36.41% | 13.59 pp | -53 | 10 | -5.30 |
| BTC Hourly | lstm | LSTM | 159 | 58 | 101 | 36.48% | 36.48% | 36.48% | 13.52 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 185 | 64 | 121 | 34.59% | 34.59% | 34.59% | 15.41 pp | -57 | 9 | -6.33 |
| BTC Hourly | xgb | XGBoost | 159 | 56 | 103 | 35.22% | 35.22% | 35.22% | 14.78 pp | -47 | 7 | -6.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 159 | 83 | 76 | 52.20% | 52.20% | 52.20% | 2.20 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 159 | 68 | 91 | 42.77% | 42.77% | 42.77% | 7.23 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 159 | 58 | 101 | 36.48% | 36.48% | 36.48% | 13.52 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 159 | 56 | 103 | 35.22% | 35.22% | 35.22% | 14.78 pp | -47 | 7 | -6.71 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 9 | -1.44 |
| BTC Daily | nn | NN | 185 | 83 | 102 | 44.86% | 44.86% | 44.86% | 5.14 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 185 | 80 | 105 | 43.24% | 43.24% | 43.24% | 6.76 pp | -25 | 9 | -2.78 |
| BTC Daily | rf | RandomForest | 185 | 73 | 112 | 39.46% | 39.46% | 39.46% | 10.54 pp | -39 | 9 | -4.33 |
| BTC Daily | xgb | XGBoost | 195 | 71 | 124 | 36.41% | 36.41% | 36.41% | 13.59 pp | -53 | 10 | -5.30 |
| BTC Daily | lstm | LSTM | 185 | 64 | 121 | 34.59% | 34.59% | 34.59% | 15.41 pp | -57 | 9 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 183 | 93 | 90 | 50.82% | 50.82% | 50.82% | 0.82 pp | 3 | 15 | 0.20 |
| BTC Market Hours | transformer | Transformer | 183 | 90 | 93 | 49.18% | 49.18% | 49.18% | 0.82 pp | -3 | 15 | -0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 15 | -1.00 |
| BTC Market Hours | rf | RandomForest | 183 | 84 | 99 | 45.90% | 45.90% | 45.90% | 4.10 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 183 | 78 | 105 | 42.62% | 42.62% | 42.62% | 7.38 pp | -27 | 15 | -1.80 |
| BTC Market Hours | lstm | LSTM | 183 | 77 | 106 | 42.08% | 42.08% | 42.08% | 7.92 pp | -29 | 15 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 183 | 95 | 88 | 51.91% | 51.91% | 51.91% | 1.91 pp | 7 | 15 | 0.47 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 183 | 89 | 94 | 48.63% | 48.63% | 48.63% | 1.37 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 183 | 86 | 97 | 46.99% | 46.99% | 46.99% | 3.01 pp | -11 | 15 | -0.73 |
| BTC Market Hours Daily | rf | RandomForest | 183 | 81 | 102 | 44.26% | 44.26% | 44.26% | 5.74 pp | -21 | 15 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 183 | 76 | 107 | 41.53% | 41.53% | 41.53% | 8.47 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 183 | 74 | 109 | 40.44% | 40.44% | 40.44% | 9.56 pp | -35 | 15 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
