# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T14:57:26.014414+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 244 | 184 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 280 | 220 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 13:00:00+00:00 | 392 | 208 | 184 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 13:00:00+00:00 | 392 | 208 | 184 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 179 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 179 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 53 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 53 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 208 | 107 | 101 | 51.44% | 51.44% | 51.44% | 1.44 pp | 6 | 16 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 208 | 107 | 101 | 51.44% | 51.44% | 51.44% | 1.44 pp | 6 | 17 | 0.35 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 184 | 93 | 91 | 50.54% | 50.54% | 50.54% | 0.54 pp | 2 | 8 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 17 | -0.12 |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| BTC Market Hours | transformer | Transformer | 208 | 102 | 106 | 49.04% | 49.04% | 49.04% | 0.96 pp | -4 | 16 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | nn | NN | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 17 | -0.47 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 208 | 97 | 111 | 46.63% | 46.63% | 46.63% | 3.37 pp | -14 | 16 | -0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| BTC Market Hours | rf | RandomForest | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 16 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 17 | -1.41 |
| BTC Hourly | transformer | Transformer | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 8 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 208 | 91 | 117 | 43.75% | 43.75% | 43.75% | 6.25 pp | -26 | 16 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Daily | nn | NN | 210 | 95 | 115 | 45.24% | 45.24% | 45.24% | 4.76 pp | -20 | 10 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 17 | -2.12 |
| BTC Market Hours | lstm | LSTM | 208 | 87 | 121 | 41.83% | 41.83% | 41.83% | 8.17 pp | -34 | 16 | -2.12 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Hourly | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |
| BTC Market Hours Daily | lstm | LSTM | 208 | 84 | 124 | 40.38% | 40.38% | 40.38% | 9.62 pp | -40 | 17 | -2.35 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 210 | 88 | 122 | 41.90% | 41.90% | 41.90% | 8.10 pp | -34 | 10 | -3.40 |
| BTC Hourly | rf | RandomForest | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 8 | -3.50 |
| BTC Daily | rf | RandomForest | 210 | 80 | 130 | 38.10% | 38.10% | 38.10% | 11.90 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 220 | 80 | 140 | 36.36% | 36.36% | 36.36% | 13.64 pp | -60 | 11 | -5.45 |
| BTC Hourly | xgb | XGBoost | 184 | 69 | 115 | 37.50% | 37.50% | 37.50% | 12.50 pp | -46 | 8 | -5.75 |
| BTC Hourly | lstm | LSTM | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 8 | -6.00 |
| BTC Daily | lstm | LSTM | 210 | 70 | 140 | 33.33% | 33.33% | 33.33% | 16.67 pp | -70 | 10 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 184 | 93 | 91 | 50.54% | 50.54% | 50.54% | 0.54 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 8 | -3.50 |
| BTC Hourly | xgb | XGBoost | 184 | 69 | 115 | 37.50% | 37.50% | 37.50% | 12.50 pp | -46 | 8 | -5.75 |
| BTC Hourly | lstm | LSTM | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 210 | 100 | 110 | 47.62% | 47.62% | 47.62% | 2.38 pp | -10 | 10 | -1.00 |
| BTC Daily | nn | NN | 210 | 95 | 115 | 45.24% | 45.24% | 45.24% | 4.76 pp | -20 | 10 | -2.00 |
| BTC Daily | transformer | Transformer | 210 | 88 | 122 | 41.90% | 41.90% | 41.90% | 8.10 pp | -34 | 10 | -3.40 |
| BTC Daily | rf | RandomForest | 210 | 80 | 130 | 38.10% | 38.10% | 38.10% | 11.90 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 220 | 80 | 140 | 36.36% | 36.36% | 36.36% | 13.64 pp | -60 | 11 | -5.45 |
| BTC Daily | lstm | LSTM | 210 | 70 | 140 | 33.33% | 33.33% | 33.33% | 16.67 pp | -70 | 10 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 208 | 107 | 101 | 51.44% | 51.44% | 51.44% | 1.44 pp | 6 | 16 | 0.38 |
| BTC Market Hours | transformer | Transformer | 208 | 102 | 106 | 49.04% | 49.04% | 49.04% | 0.96 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 208 | 97 | 111 | 46.63% | 46.63% | 46.63% | 3.37 pp | -14 | 16 | -0.88 |
| BTC Market Hours | rf | RandomForest | 208 | 95 | 113 | 45.67% | 45.67% | 45.67% | 4.33 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 208 | 91 | 117 | 43.75% | 43.75% | 43.75% | 6.25 pp | -26 | 16 | -1.62 |
| BTC Market Hours | lstm | LSTM | 208 | 87 | 121 | 41.83% | 41.83% | 41.83% | 8.17 pp | -34 | 16 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 208 | 107 | 101 | 51.44% | 51.44% | 51.44% | 1.44 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 208 | 103 | 105 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 17 | -0.12 |
| BTC Market Hours Daily | nn | NN | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 17 | -0.47 |
| BTC Market Hours Daily | rf | RandomForest | 208 | 92 | 116 | 44.23% | 44.23% | 44.23% | 5.77 pp | -24 | 17 | -1.41 |
| BTC Market Hours Daily | xgb | XGBoost | 208 | 86 | 122 | 41.35% | 41.35% | 41.35% | 8.65 pp | -36 | 17 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 208 | 84 | 124 | 40.38% | 40.38% | 40.38% | 9.62 pp | -40 | 17 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
