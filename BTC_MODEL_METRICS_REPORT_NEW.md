# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T17:37:45.514820+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 263 | 203 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 298 | 238 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 16:00:00+00:00 | 426 | 226 | 200 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 16:00:00+00:00 | 426 | 226 | 200 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 226 | 118 | 108 | 52.21% | 52.21% | 52.21% | 2.21 pp | 10 | 18 | 0.56 |
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Market Hours Daily | transformer | Transformer | 226 | 113 | 113 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 19 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 19 | -0.42 |
| BTC Market Hours Daily | nn | NN | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 226 | 107 | 119 | 47.35% | 47.35% | 47.35% | 2.65 pp | -12 | 18 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 226 | 106 | 120 | 46.90% | 46.90% | 46.90% | 3.10 pp | -14 | 18 | -0.78 |
| BTC Market Hours | rf | RandomForest | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 226 | 100 | 126 | 44.25% | 44.25% | 44.25% | 5.75 pp | -26 | 19 | -1.37 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | xgb | XGBoost | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 19 | -1.89 |
| BTC Daily | mlp_sklearn | MLPClassifier | 228 | 104 | 124 | 45.61% | 45.61% | 45.61% | 4.39 pp | -20 | 10 | -2.00 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 226 | 93 | 133 | 41.15% | 41.15% | 41.15% | 8.85 pp | -40 | 18 | -2.22 |
| BTC Hourly | transformer | Transformer | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 226 | 90 | 136 | 39.82% | 39.82% | 39.82% | 10.18 pp | -46 | 19 | -2.42 |
| BTC Daily | nn | NN | 228 | 101 | 127 | 44.30% | 44.30% | 44.30% | 5.70 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 203 | 85 | 118 | 41.87% | 41.87% | 41.87% | 8.13 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 228 | 90 | 138 | 39.47% | 39.47% | 39.47% | 10.53 pp | -48 | 10 | -4.80 |
| BTC Daily | rf | RandomForest | 228 | 87 | 141 | 38.16% | 38.16% | 38.16% | 11.84 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 203 | 76 | 127 | 37.44% | 37.44% | 37.44% | 12.56 pp | -51 | 9 | -5.67 |
| BTC Daily | xgb | XGBoost | 238 | 82 | 156 | 34.45% | 34.45% | 34.45% | 15.55 pp | -74 | 11 | -6.73 |
| BTC Hourly | xgb | XGBoost | 203 | 71 | 132 | 34.98% | 34.98% | 34.98% | 15.02 pp | -61 | 9 | -6.78 |
| BTC Daily | lstm | LSTM | 228 | 76 | 152 | 33.33% | 33.33% | 33.33% | 16.67 pp | -76 | 10 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 203 | 85 | 118 | 41.87% | 41.87% | 41.87% | 8.13 pp | -33 | 9 | -3.67 |
| BTC Hourly | lstm | LSTM | 203 | 76 | 127 | 37.44% | 37.44% | 37.44% | 12.56 pp | -51 | 9 | -5.67 |
| BTC Hourly | xgb | XGBoost | 203 | 71 | 132 | 34.98% | 34.98% | 34.98% | 15.02 pp | -61 | 9 | -6.78 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 228 | 104 | 124 | 45.61% | 45.61% | 45.61% | 4.39 pp | -20 | 10 | -2.00 |
| BTC Daily | nn | NN | 228 | 101 | 127 | 44.30% | 44.30% | 44.30% | 5.70 pp | -26 | 10 | -2.60 |
| BTC Daily | transformer | Transformer | 228 | 90 | 138 | 39.47% | 39.47% | 39.47% | 10.53 pp | -48 | 10 | -4.80 |
| BTC Daily | rf | RandomForest | 228 | 87 | 141 | 38.16% | 38.16% | 38.16% | 11.84 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 238 | 82 | 156 | 34.45% | 34.45% | 34.45% | 15.55 pp | -74 | 11 | -6.73 |
| BTC Daily | lstm | LSTM | 228 | 76 | 152 | 33.33% | 33.33% | 33.33% | 16.67 pp | -76 | 10 | -7.60 |

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
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
