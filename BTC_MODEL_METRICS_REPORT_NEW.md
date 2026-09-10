# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T18:58:53.131399+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 311 | 251 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 347 | 287 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 17:00:00+00:00 | 515 | 275 | 240 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 17:00:00+00:00 | 515 | 275 | 240 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 241 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 241 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 241 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 242 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 275 | 141 | 134 | 51.27% | 51.25% | 51.27% | 1.27 pp | 7 | 22 | 0.32 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 275 | 135 | 140 | 49.09% | 48.33% | 49.09% | 0.91 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 275 | 134 | 141 | 48.73% | 48.33% | 48.73% | 1.27 pp | -7 | 23 | -0.30 |
| BTC Market Hours Daily | nn | NN | 275 | 132 | 143 | 48.00% | 48.33% | 48.00% | 2.00 pp | -11 | 23 | -0.48 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 251 | 122 | 129 | 48.61% | 48.33% | 48.61% | 1.39 pp | -7 | 11 | -0.64 |
| BTC Market Hours | transformer | Transformer | 275 | 128 | 147 | 46.55% | 45.83% | 46.55% | 3.45 pp | -19 | 22 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 275 | 127 | 148 | 46.18% | 46.25% | 46.18% | 3.82 pp | -21 | 22 | -0.95 |
| Consolidated Hourly | rf | RandomForest | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 275 | 124 | 151 | 45.09% | 45.00% | 45.09% | 4.91 pp | -27 | 22 | -1.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 275 | 121 | 154 | 44.00% | 42.50% | 44.00% | 6.00 pp | -33 | 22 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 241 | 109 | 132 | 45.23% | 45.00% | 45.23% | 4.77 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 109 | 132 | 45.23% | 45.00% | 45.23% | 4.77 pp | -23 | 15 | -1.53 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 275 | 119 | 156 | 43.27% | 42.50% | 43.27% | 6.73 pp | -37 | 23 | -1.61 |
| BTC Market Hours Daily | rf | RandomForest | 275 | 118 | 157 | 42.91% | 42.08% | 42.91% | 7.09 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| BTC Daily | nn | NN | 277 | 127 | 150 | 45.85% | 45.42% | 45.85% | 4.15 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | transformer | Transformer | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 275 | 113 | 162 | 41.09% | 42.50% | 41.09% | 8.91 pp | -49 | 23 | -2.13 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| BTC Market Hours | lstm | LSTM | 275 | 113 | 162 | 41.09% | 42.08% | 41.09% | 8.91 pp | -49 | 22 | -2.23 |
| BTC Daily | mlp_sklearn | MLPClassifier | 277 | 125 | 152 | 45.13% | 45.00% | 45.13% | 4.87 pp | -27 | 12 | -2.25 |
| BTC Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 45.83% | 45.02% | 4.98 pp | -25 | 11 | -2.27 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 241 | 101 | 140 | 41.91% | 42.08% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 101 | 140 | 41.91% | 42.08% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 34 | 53 | 39.08% | 39.08% | 39.08% | 10.92 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 251 | 105 | 146 | 41.83% | 42.08% | 41.83% | 8.17 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 277 | 112 | 165 | 40.43% | 38.33% | 40.43% | 9.57 pp | -53 | 12 | -4.42 |
| BTC Daily | xgb | XGBoost | 287 | 107 | 180 | 37.28% | 37.50% | 37.28% | 12.72 pp | -73 | 13 | -5.62 |
| BTC Daily | rf | RandomForest | 277 | 104 | 173 | 37.55% | 37.08% | 37.55% | 12.45 pp | -69 | 12 | -5.75 |
| BTC Hourly | lstm | LSTM | 251 | 92 | 159 | 36.65% | 36.25% | 36.65% | 13.35 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 251 | 88 | 163 | 35.06% | 35.83% | 35.06% | 14.94 pp | -75 | 11 | -6.82 |
| BTC Daily | lstm | LSTM | 277 | 97 | 180 | 35.02% | 35.42% | 35.02% | 14.98 pp | -83 | 12 | -6.92 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 251 | 122 | 129 | 48.61% | 48.33% | 48.61% | 1.39 pp | -7 | 11 | -0.64 |
| BTC Hourly | transformer | Transformer | 251 | 113 | 138 | 45.02% | 45.83% | 45.02% | 4.98 pp | -25 | 11 | -2.27 |
| BTC Hourly | nn | NN | 251 | 105 | 146 | 41.83% | 42.08% | 41.83% | 8.17 pp | -41 | 11 | -3.73 |
| BTC Hourly | rf | RandomForest | 251 | 103 | 148 | 41.04% | 41.67% | 41.04% | 8.96 pp | -45 | 11 | -4.09 |
| BTC Hourly | lstm | LSTM | 251 | 92 | 159 | 36.65% | 36.25% | 36.65% | 13.35 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 251 | 88 | 163 | 35.06% | 35.83% | 35.06% | 14.94 pp | -75 | 11 | -6.82 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 277 | 127 | 150 | 45.85% | 45.42% | 45.85% | 4.15 pp | -23 | 12 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 277 | 125 | 152 | 45.13% | 45.00% | 45.13% | 4.87 pp | -27 | 12 | -2.25 |
| BTC Daily | transformer | Transformer | 277 | 112 | 165 | 40.43% | 38.33% | 40.43% | 9.57 pp | -53 | 12 | -4.42 |
| BTC Daily | xgb | XGBoost | 287 | 107 | 180 | 37.28% | 37.50% | 37.28% | 12.72 pp | -73 | 13 | -5.62 |
| BTC Daily | rf | RandomForest | 277 | 104 | 173 | 37.55% | 37.08% | 37.55% | 12.45 pp | -69 | 12 | -5.75 |
| BTC Daily | lstm | LSTM | 277 | 97 | 180 | 35.02% | 35.42% | 35.02% | 14.98 pp | -83 | 12 | -6.92 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 275 | 141 | 134 | 51.27% | 51.25% | 51.27% | 1.27 pp | 7 | 22 | 0.32 |
| BTC Market Hours | transformer | Transformer | 275 | 128 | 147 | 46.55% | 45.83% | 46.55% | 3.45 pp | -19 | 22 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 275 | 127 | 148 | 46.18% | 46.25% | 46.18% | 3.82 pp | -21 | 22 | -0.95 |
| BTC Market Hours | xgb | XGBoost | 275 | 124 | 151 | 45.09% | 45.00% | 45.09% | 4.91 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 275 | 121 | 154 | 44.00% | 42.50% | 44.00% | 6.00 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 275 | 113 | 162 | 41.09% | 42.08% | 41.09% | 8.91 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 275 | 135 | 140 | 49.09% | 48.33% | 49.09% | 0.91 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 275 | 134 | 141 | 48.73% | 48.33% | 48.73% | 1.27 pp | -7 | 23 | -0.30 |
| BTC Market Hours Daily | nn | NN | 275 | 132 | 143 | 48.00% | 48.33% | 48.00% | 2.00 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | xgb | XGBoost | 275 | 119 | 156 | 43.27% | 42.50% | 43.27% | 6.73 pp | -37 | 23 | -1.61 |
| BTC Market Hours Daily | rf | RandomForest | 275 | 118 | 157 | 42.91% | 42.08% | 42.91% | 7.09 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | lstm | LSTM | 275 | 113 | 162 | 41.09% | 42.50% | 41.09% | 8.91 pp | -49 | 23 | -2.13 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 15 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | lstm | LSTM | 241 | 109 | 132 | 45.23% | 45.00% | 45.23% | 4.77 pp | -23 | 15 | -1.53 |
| Consolidated Hourly | transformer | Transformer | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | xgb | XGBoost | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Hourly | nn | NN | 241 | 101 | 140 | 41.91% | 42.08% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 15 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 241 | 111 | 130 | 46.06% | 46.25% | 46.06% | 3.94 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 241 | 109 | 132 | 45.23% | 45.00% | 45.23% | 4.77 pp | -23 | 15 | -1.53 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 241 | 105 | 136 | 43.57% | 43.33% | 43.57% | 6.43 pp | -31 | 15 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 241 | 101 | 140 | 41.91% | 42.08% | 41.91% | 8.09 pp | -39 | 15 | -2.60 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 34 | 53 | 39.08% | 39.08% | 39.08% | 10.92 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
