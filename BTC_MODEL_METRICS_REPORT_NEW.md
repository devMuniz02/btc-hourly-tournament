# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T16:42:38.490768+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 15:00:00+00:00 | 425 | 226 | 199 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 196 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 196 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 196 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 197 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 226 | 118 | 108 | 52.21% | 52.21% | 52.21% | 2.21 pp | 10 | 18 | 0.56 |
| BTC Market Hours Daily | transformer | Transformer | 226 | 113 | 113 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 202 | 101 | 101 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 19 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 18 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 226 | 106 | 120 | 46.90% | 46.90% | 46.90% | 3.10 pp | -14 | 18 | -0.78 |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 5 | -0.80 |
| BTC Market Hours | rf | RandomForest | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 13 | -1.23 |
| BTC Market Hours | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 19 | -1.37 |
| Consolidated Hourly | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 13 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | lstm | LSTM | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 19 | -1.89 |
| BTC Daily | mlp_sklearn | MLPClassifier | 228 | 104 | 124 | 45.61% | 45.61% | 45.61% | 4.39 pp | -20 | 10 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 196 | 85 | 111 | 43.37% | 43.37% | 43.37% | 6.63 pp | -26 | 13 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 196 | 85 | 111 | 43.37% | 43.37% | 43.37% | 6.63 pp | -26 | 13 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 202 | 91 | 111 | 45.05% | 45.05% | 45.05% | 4.95 pp | -20 | 9 | -2.22 |
| BTC Market Hours | lstm | LSTM | 226 | 93 | 133 | 41.15% | 41.15% | 41.15% | 8.85 pp | -40 | 18 | -2.22 |
| BTC Daily | nn | NN | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 226 | 90 | 136 | 39.82% | 39.82% | 39.82% | 10.18 pp | -46 | 19 | -2.42 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
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
| BTC Market Hours Daily | transformer | Transformer | 226 | 113 | 113 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 19 | -0.42 |
| BTC Market Hours Daily | nn | NN | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 19 | -0.63 |
| BTC Market Hours Daily | rf | RandomForest | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 19 | -1.37 |
| BTC Market Hours Daily | xgb | XGBoost | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 19 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 226 | 90 | 136 | 39.82% | 39.82% | 39.82% | 10.18 pp | -46 | 19 | -2.42 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | rf | RandomForest | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Hourly | xgb | XGBoost | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 13 | -1.23 |
| Consolidated Hourly | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 13 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | transformer | Transformer | 196 | 85 | 111 | 43.37% | 43.37% | 43.37% | 6.63 pp | -26 | 13 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 196 | 96 | 100 | 48.98% | 48.98% | 48.98% | 1.02 pp | -4 | 13 | -0.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 196 | 90 | 106 | 45.92% | 45.92% | 45.92% | 4.08 pp | -16 | 13 | -1.23 |
| Consolidated Daily/Hourly Refresh | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 13 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 196 | 87 | 109 | 44.39% | 44.39% | 44.39% | 5.61 pp | -22 | 13 | -1.69 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 196 | 85 | 111 | 43.37% | 43.37% | 43.37% | 6.63 pp | -26 | 13 | -2.00 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
