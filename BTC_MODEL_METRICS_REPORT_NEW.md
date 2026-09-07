# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T16:23:10.321628+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 262 | 202 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 298 | 238 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 15:00:00+00:00 | 425 | 226 | 199 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 15:00:00+00:00 | 424 | 225 | 199 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 195 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 19:00:00+00:00 | 195 | 61 | 134 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 226 | 118 | 108 | 52.21% | 52.21% | 52.21% | 2.21 pp | 10 | 18 | 0.56 |
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 225 | 112 | 113 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 19 | -0.47 |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | nn | NN | 225 | 106 | 119 | 47.11% | 47.11% | 47.11% | 2.89 pp | -13 | 19 | -0.68 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 226 | 106 | 120 | 46.90% | 46.90% | 46.90% | 3.10 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 18 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 19 | -1.32 |
| BTC Market Hours | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 18 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 19 | -1.95 |
| BTC Daily | mlp_sklearn | MLPClassifier | 228 | 104 | 124 | 45.61% | 45.61% | 45.61% | 4.39 pp | -20 | 10 | -2.00 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 9 | -2.22 |
| BTC Market Hours | lstm | LSTM | 226 | 93 | 133 | 41.15% | 41.15% | 41.15% | 8.85 pp | -40 | 18 | -2.22 |
| BTC Daily | nn | NN | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 225 | 89 | 136 | 39.56% | 39.56% | 39.56% | 10.44 pp | -47 | 19 | -2.47 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 202 | 86 | 116 | 42.57% | 42.57% | 42.57% | 7.43 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 202 | 85 | 117 | 42.08% | 42.08% | 42.08% | 7.92 pp | -32 | 9 | -3.56 |
| BTC Daily | transformer | Transformer | 228 | 91 | 137 | 39.91% | 39.91% | 39.91% | 10.09 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 228 | 88 | 140 | 38.60% | 38.60% | 38.60% | 11.40 pp | -52 | 10 | -5.20 |
| BTC Hourly | lstm | LSTM | 202 | 76 | 126 | 37.62% | 37.62% | 37.62% | 12.38 pp | -50 | 9 | -5.56 |
| BTC Daily | xgb | XGBoost | 238 | 83 | 155 | 34.87% | 34.87% | 34.87% | 15.13 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 202 | 71 | 131 | 35.15% | 35.15% | 35.15% | 14.85 pp | -60 | 9 | -6.67 |
| BTC Daily | lstm | LSTM | 228 | 75 | 153 | 32.89% | 32.89% | 32.89% | 17.11 pp | -78 | 10 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 9 | -2.22 |
| BTC Hourly | nn | NN | 202 | 86 | 116 | 42.57% | 42.57% | 42.57% | 7.43 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 202 | 85 | 117 | 42.08% | 42.08% | 42.08% | 7.92 pp | -32 | 9 | -3.56 |
| BTC Hourly | lstm | LSTM | 202 | 76 | 126 | 37.62% | 37.62% | 37.62% | 12.38 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 202 | 71 | 131 | 35.15% | 35.15% | 35.15% | 14.85 pp | -60 | 9 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 228 | 104 | 124 | 45.61% | 45.61% | 45.61% | 4.39 pp | -20 | 10 | -2.00 |
| BTC Daily | nn | NN | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 228 | 91 | 137 | 39.91% | 39.91% | 39.91% | 10.09 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 228 | 88 | 140 | 38.60% | 38.60% | 38.60% | 11.40 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 238 | 83 | 155 | 34.87% | 34.87% | 34.87% | 15.13 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 228 | 75 | 153 | 32.89% | 32.89% | 32.89% | 17.11 pp | -78 | 10 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 226 | 118 | 108 | 52.21% | 52.21% | 52.21% | 2.21 pp | 10 | 18 | 0.56 |
| BTC Market Hours | transformer | Transformer | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 18 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 226 | 106 | 120 | 46.90% | 46.90% | 46.90% | 3.10 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 18 | -1.33 |
| BTC Market Hours | lstm | LSTM | 226 | 93 | 133 | 41.15% | 41.15% | 41.15% | 8.85 pp | -40 | 18 | -2.22 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 225 | 112 | 113 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | nn | NN | 225 | 106 | 119 | 47.11% | 47.11% | 47.11% | 2.89 pp | -13 | 19 | -0.68 |
| BTC Market Hours Daily | rf | RandomForest | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 19 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 225 | 89 | 136 | 39.56% | 39.56% | 39.56% | 10.44 pp | -47 | 19 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 98 | 97 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 86 | 109 | 44.10% | 44.10% | 44.10% | 5.90 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
