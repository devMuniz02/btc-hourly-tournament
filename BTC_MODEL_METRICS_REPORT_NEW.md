# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T16:22:43.222496+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 294 | 234 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 330 | 270 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 15:00:00+00:00 | 483 | 258 | 225 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 15:00:00+00:00 | 483 | 258 | 225 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 226 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T23:00:00+00:00 | 227 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 258 | 136 | 122 | 52.71% | 52.50% | 52.71% | 2.71 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 258 | 129 | 129 | 50.00% | 49.17% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 258 | 127 | 131 | 49.22% | 49.17% | 49.22% | 0.78 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 258 | 125 | 133 | 48.45% | 48.33% | 48.45% | 1.55 pp | -8 | 21 | -0.38 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 258 | 122 | 136 | 47.29% | 47.50% | 47.29% | 2.71 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 258 | 121 | 137 | 46.90% | 46.67% | 46.90% | 3.10 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 258 | 119 | 139 | 46.12% | 44.58% | 46.12% | 3.88 pp | -20 | 20 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| BTC Market Hours | rf | RandomForest | 258 | 116 | 142 | 44.96% | 43.75% | 44.96% | 5.04 pp | -26 | 20 | -1.30 |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 258 | 114 | 144 | 44.19% | 43.75% | 44.19% | 5.81 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 258 | 113 | 145 | 43.80% | 42.50% | 43.80% | 6.20 pp | -32 | 21 | -1.52 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| BTC Daily | mlp_sklearn | MLPClassifier | 260 | 120 | 140 | 46.15% | 45.83% | 46.15% | 3.85 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| BTC Daily | nn | NN | 260 | 118 | 142 | 45.38% | 44.58% | 45.38% | 4.62 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 258 | 107 | 151 | 41.47% | 41.67% | 41.47% | 8.53 pp | -44 | 21 | -2.10 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| BTC Market Hours | lstm | LSTM | 258 | 105 | 153 | 40.70% | 41.67% | 40.70% | 9.30 pp | -48 | 20 | -2.40 |
| Consolidated Hourly | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| BTC Hourly | transformer | Transformer | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 10 | -2.80 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 29 | 50 | 36.71% | 36.71% | 36.71% | 13.29 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |
| BTC Hourly | nn | NN | 234 | 98 | 136 | 41.88% | 41.88% | 41.88% | 8.12 pp | -38 | 10 | -3.80 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 6 | -3.83 |
| BTC Daily | transformer | Transformer | 260 | 105 | 155 | 40.38% | 39.17% | 40.38% | 9.62 pp | -50 | 12 | -4.17 |
| BTC Hourly | rf | RandomForest | 234 | 96 | 138 | 41.03% | 41.03% | 41.03% | 8.97 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 260 | 97 | 163 | 37.31% | 36.67% | 37.31% | 12.69 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 270 | 97 | 173 | 35.93% | 35.42% | 35.93% | 14.07 pp | -76 | 13 | -5.85 |
| BTC Hourly | lstm | LSTM | 234 | 87 | 147 | 37.18% | 37.18% | 37.18% | 12.82 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 260 | 92 | 168 | 35.38% | 35.83% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 234 | 80 | 154 | 34.19% | 34.19% | 34.19% | 15.81 pp | -74 | 10 | -7.40 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 234 | 115 | 119 | 49.15% | 49.15% | 49.15% | 0.85 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 234 | 103 | 131 | 44.02% | 44.02% | 44.02% | 5.98 pp | -28 | 10 | -2.80 |
| BTC Hourly | nn | NN | 234 | 98 | 136 | 41.88% | 41.88% | 41.88% | 8.12 pp | -38 | 10 | -3.80 |
| BTC Hourly | rf | RandomForest | 234 | 96 | 138 | 41.03% | 41.03% | 41.03% | 8.97 pp | -42 | 10 | -4.20 |
| BTC Hourly | lstm | LSTM | 234 | 87 | 147 | 37.18% | 37.18% | 37.18% | 12.82 pp | -60 | 10 | -6.00 |
| BTC Hourly | xgb | XGBoost | 234 | 80 | 154 | 34.19% | 34.19% | 34.19% | 15.81 pp | -74 | 10 | -7.40 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 260 | 120 | 140 | 46.15% | 45.83% | 46.15% | 3.85 pp | -20 | 12 | -1.67 |
| BTC Daily | nn | NN | 260 | 118 | 142 | 45.38% | 44.58% | 45.38% | 4.62 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 260 | 105 | 155 | 40.38% | 39.17% | 40.38% | 9.62 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 260 | 97 | 163 | 37.31% | 36.67% | 37.31% | 12.69 pp | -66 | 12 | -5.50 |
| BTC Daily | xgb | XGBoost | 270 | 97 | 173 | 35.93% | 35.42% | 35.93% | 14.07 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 260 | 92 | 168 | 35.38% | 35.83% | 35.38% | 14.62 pp | -76 | 12 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 258 | 136 | 122 | 52.71% | 52.50% | 52.71% | 2.71 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 258 | 122 | 136 | 47.29% | 47.50% | 47.29% | 2.71 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 258 | 121 | 137 | 46.90% | 46.67% | 46.90% | 3.10 pp | -16 | 20 | -0.80 |
| BTC Market Hours | xgb | XGBoost | 258 | 119 | 139 | 46.12% | 44.58% | 46.12% | 3.88 pp | -20 | 20 | -1.00 |
| BTC Market Hours | rf | RandomForest | 258 | 116 | 142 | 44.96% | 43.75% | 44.96% | 5.04 pp | -26 | 20 | -1.30 |
| BTC Market Hours | lstm | LSTM | 258 | 105 | 153 | 40.70% | 41.67% | 40.70% | 9.30 pp | -48 | 20 | -2.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 258 | 129 | 129 | 50.00% | 49.17% | 50.00% | 0.00 pp | 0 | 21 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 258 | 127 | 131 | 49.22% | 49.17% | 49.22% | 0.78 pp | -4 | 21 | -0.19 |
| BTC Market Hours Daily | nn | NN | 258 | 125 | 133 | 48.45% | 48.33% | 48.45% | 1.55 pp | -8 | 21 | -0.38 |
| BTC Market Hours Daily | xgb | XGBoost | 258 | 114 | 144 | 44.19% | 43.75% | 44.19% | 5.81 pp | -30 | 21 | -1.43 |
| BTC Market Hours Daily | rf | RandomForest | 258 | 113 | 145 | 43.80% | 42.50% | 43.80% | 6.20 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 258 | 107 | 151 | 41.47% | 41.67% | 41.47% | 8.53 pp | -44 | 21 | -2.10 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Hourly | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Hourly | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 226 | 109 | 117 | 48.23% | 48.23% | 48.23% | 1.77 pp | -8 | 14 | -0.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 226 | 104 | 122 | 46.02% | 46.02% | 46.02% | 3.98 pp | -18 | 14 | -1.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 226 | 102 | 124 | 45.13% | 45.13% | 45.13% | 4.87 pp | -22 | 14 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 226 | 101 | 125 | 44.69% | 44.69% | 44.69% | 5.31 pp | -24 | 14 | -1.71 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 226 | 96 | 130 | 42.48% | 42.48% | 42.48% | 7.52 pp | -34 | 14 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 226 | 95 | 131 | 42.04% | 42.04% | 42.04% | 7.96 pp | -36 | 14 | -2.57 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | xgb | XGBoost | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 78 | 29 | 49 | 37.18% | 37.18% | 37.18% | 12.82 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | nn | NN | 78 | 28 | 50 | 35.90% | 35.90% | 35.90% | 14.10 pp | -22 | 6 | -3.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 79 | 36 | 43 | 45.57% | 45.57% | 45.57% | 4.43 pp | -7 | 6 | -1.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | xgb | XGBoost | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 79 | 29 | 50 | 36.71% | 36.71% | 36.71% | 13.29 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 79 | 28 | 51 | 35.44% | 35.44% | 35.44% | 14.56 pp | -23 | 6 | -3.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
