# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-10T20:21:30.067557+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 312 | 252 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 348 | 288 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-10 19:00:00+00:00 | 518 | 276 | 242 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-10 19:00:00+00:00 | 518 | 276 | 242 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 242 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 242 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 242 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-01T18:00:00+00:00 | 243 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 276 | 141 | 135 | 51.09% | 51.25% | 51.09% | 1.09 pp | 6 | 22 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 276 | 135 | 141 | 48.91% | 48.33% | 48.91% | 1.09 pp | -6 | 23 | -0.26 |
| BTC Market Hours Daily | transformer | Transformer | 276 | 134 | 142 | 48.55% | 47.92% | 48.55% | 1.45 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | nn | NN | 276 | 132 | 144 | 47.83% | 48.33% | 47.83% | 2.17 pp | -12 | 23 | -0.52 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 252 | 122 | 130 | 48.41% | 47.92% | 48.41% | 1.59 pp | -8 | 11 | -0.73 |
| BTC Market Hours | transformer | Transformer | 276 | 128 | 148 | 46.38% | 45.83% | 46.38% | 3.62 pp | -20 | 22 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 276 | 127 | 149 | 46.01% | 46.25% | 46.01% | 3.99 pp | -22 | 22 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 15 | -1.07 |
| BTC Market Hours | xgb | XGBoost | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 22 | -1.18 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 242 | 111 | 131 | 45.87% | 45.83% | 45.87% | 4.13 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 242 | 111 | 131 | 45.87% | 45.83% | 45.87% | 4.13 pp | -20 | 15 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 86 | 38 | 48 | 44.19% | 44.19% | 44.19% | 5.81 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 276 | 122 | 154 | 44.20% | 42.92% | 44.20% | 5.80 pp | -32 | 22 | -1.45 |
| Consolidated Hourly | lstm | LSTM | 242 | 110 | 132 | 45.45% | 45.42% | 45.45% | 4.55 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 242 | 110 | 132 | 45.45% | 45.42% | 45.45% | 4.55 pp | -22 | 15 | -1.47 |
| Consolidated Market Hours Daily | rf | RandomForest | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Market Hours Daily | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 276 | 119 | 157 | 43.12% | 42.50% | 43.12% | 6.88 pp | -38 | 23 | -1.65 |
| Consolidated Market Hours | rf | RandomForest | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 276 | 118 | 158 | 42.75% | 42.08% | 42.75% | 7.25 pp | -40 | 23 | -1.74 |
| BTC Daily | nn | NN | 278 | 127 | 151 | 45.68% | 45.00% | 45.68% | 4.32 pp | -24 | 12 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 242 | 106 | 136 | 43.80% | 43.75% | 43.80% | 6.20 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 242 | 106 | 136 | 43.80% | 43.75% | 43.80% | 6.20 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 242 | 105 | 137 | 43.39% | 43.33% | 43.39% | 6.61 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 242 | 105 | 137 | 43.39% | 43.33% | 43.39% | 6.61 pp | -32 | 15 | -2.13 |
| Consolidated Market Hours Daily | xgb | XGBoost | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| BTC Daily | mlp_sklearn | MLPClassifier | 278 | 126 | 152 | 45.32% | 45.00% | 45.32% | 4.68 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 276 | 113 | 163 | 40.94% | 42.50% | 40.94% | 9.06 pp | -50 | 23 | -2.17 |
| BTC Hourly | transformer | Transformer | 252 | 114 | 138 | 45.24% | 45.83% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| BTC Market Hours | lstm | LSTM | 276 | 114 | 162 | 41.30% | 42.50% | 41.30% | 8.70 pp | -48 | 22 | -2.18 |
| Consolidated Market Hours | xgb | XGBoost | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 242 | 101 | 141 | 41.74% | 41.67% | 41.74% | 8.26 pp | -40 | 15 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 242 | 101 | 141 | 41.74% | 41.67% | 41.74% | 8.26 pp | -40 | 15 | -2.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 87 | 34 | 53 | 39.08% | 39.08% | 39.08% | 10.92 pp | -19 | 7 | -2.71 |
| Consolidated Market Hours | lstm | LSTM | 86 | 33 | 53 | 38.37% | 38.37% | 38.37% | 11.63 pp | -20 | 7 | -2.86 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 86 | 32 | 54 | 37.21% | 37.21% | 37.21% | 12.79 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 7 | -3.29 |
| Consolidated Market Hours | nn | NN | 86 | 31 | 55 | 36.05% | 36.05% | 36.05% | 13.95 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 252 | 105 | 147 | 41.67% | 41.67% | 41.67% | 8.33 pp | -42 | 11 | -3.82 |
| BTC Hourly | rf | RandomForest | 252 | 104 | 148 | 41.27% | 42.08% | 41.27% | 8.73 pp | -44 | 11 | -4.00 |
| BTC Daily | transformer | Transformer | 278 | 112 | 166 | 40.29% | 37.92% | 40.29% | 9.71 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 278 | 105 | 173 | 37.77% | 37.08% | 37.77% | 12.23 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 288 | 107 | 181 | 37.15% | 37.08% | 37.15% | 12.85 pp | -74 | 13 | -5.69 |
| BTC Hourly | lstm | LSTM | 252 | 93 | 159 | 36.90% | 36.67% | 36.90% | 13.10 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 252 | 88 | 164 | 34.92% | 35.42% | 34.92% | 15.08 pp | -76 | 11 | -6.91 |
| BTC Daily | lstm | LSTM | 278 | 97 | 181 | 34.89% | 35.00% | 34.89% | 15.11 pp | -84 | 12 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 252 | 122 | 130 | 48.41% | 47.92% | 48.41% | 1.59 pp | -8 | 11 | -0.73 |
| BTC Hourly | transformer | Transformer | 252 | 114 | 138 | 45.24% | 45.83% | 45.24% | 4.76 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 252 | 105 | 147 | 41.67% | 41.67% | 41.67% | 8.33 pp | -42 | 11 | -3.82 |
| BTC Hourly | rf | RandomForest | 252 | 104 | 148 | 41.27% | 42.08% | 41.27% | 8.73 pp | -44 | 11 | -4.00 |
| BTC Hourly | lstm | LSTM | 252 | 93 | 159 | 36.90% | 36.67% | 36.90% | 13.10 pp | -66 | 11 | -6.00 |
| BTC Hourly | xgb | XGBoost | 252 | 88 | 164 | 34.92% | 35.42% | 34.92% | 15.08 pp | -76 | 11 | -6.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 278 | 127 | 151 | 45.68% | 45.00% | 45.68% | 4.32 pp | -24 | 12 | -2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 278 | 126 | 152 | 45.32% | 45.00% | 45.32% | 4.68 pp | -26 | 12 | -2.17 |
| BTC Daily | transformer | Transformer | 278 | 112 | 166 | 40.29% | 37.92% | 40.29% | 9.71 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 278 | 105 | 173 | 37.77% | 37.08% | 37.77% | 12.23 pp | -68 | 12 | -5.67 |
| BTC Daily | xgb | XGBoost | 288 | 107 | 181 | 37.15% | 37.08% | 37.15% | 12.85 pp | -74 | 13 | -5.69 |
| BTC Daily | lstm | LSTM | 278 | 97 | 181 | 34.89% | 35.00% | 34.89% | 15.11 pp | -84 | 12 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 276 | 141 | 135 | 51.09% | 51.25% | 51.09% | 1.09 pp | 6 | 22 | 0.27 |
| BTC Market Hours | transformer | Transformer | 276 | 128 | 148 | 46.38% | 45.83% | 46.38% | 3.62 pp | -20 | 22 | -0.91 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 276 | 127 | 149 | 46.01% | 46.25% | 46.01% | 3.99 pp | -22 | 22 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 276 | 125 | 151 | 45.29% | 45.00% | 45.29% | 4.71 pp | -26 | 22 | -1.18 |
| BTC Market Hours | rf | RandomForest | 276 | 122 | 154 | 44.20% | 42.92% | 44.20% | 5.80 pp | -32 | 22 | -1.45 |
| BTC Market Hours | lstm | LSTM | 276 | 114 | 162 | 41.30% | 42.50% | 41.30% | 8.70 pp | -48 | 22 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 276 | 135 | 141 | 48.91% | 48.33% | 48.91% | 1.09 pp | -6 | 23 | -0.26 |
| BTC Market Hours Daily | transformer | Transformer | 276 | 134 | 142 | 48.55% | 47.92% | 48.55% | 1.45 pp | -8 | 23 | -0.35 |
| BTC Market Hours Daily | nn | NN | 276 | 132 | 144 | 47.83% | 48.33% | 47.83% | 2.17 pp | -12 | 23 | -0.52 |
| BTC Market Hours Daily | xgb | XGBoost | 276 | 119 | 157 | 43.12% | 42.50% | 43.12% | 6.88 pp | -38 | 23 | -1.65 |
| BTC Market Hours Daily | rf | RandomForest | 276 | 118 | 158 | 42.75% | 42.08% | 42.75% | 7.25 pp | -40 | 23 | -1.74 |
| BTC Market Hours Daily | lstm | LSTM | 276 | 113 | 163 | 40.94% | 42.50% | 40.94% | 9.06 pp | -50 | 23 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 15 | -1.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 242 | 111 | 131 | 45.87% | 45.83% | 45.87% | 4.13 pp | -20 | 15 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 242 | 110 | 132 | 45.45% | 45.42% | 45.45% | 4.55 pp | -22 | 15 | -1.47 |
| Consolidated Hourly | transformer | Transformer | 242 | 106 | 136 | 43.80% | 43.75% | 43.80% | 6.20 pp | -30 | 15 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 242 | 105 | 137 | 43.39% | 43.33% | 43.39% | 6.61 pp | -32 | 15 | -2.13 |
| Consolidated Hourly | nn | NN | 242 | 101 | 141 | 41.74% | 41.67% | 41.74% | 8.26 pp | -40 | 15 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 15 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 242 | 111 | 131 | 45.87% | 45.83% | 45.87% | 4.13 pp | -20 | 15 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 242 | 110 | 132 | 45.45% | 45.42% | 45.45% | 4.55 pp | -22 | 15 | -1.47 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 242 | 106 | 136 | 43.80% | 43.75% | 43.80% | 6.20 pp | -30 | 15 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 242 | 105 | 137 | 43.39% | 43.33% | 43.39% | 6.61 pp | -32 | 15 | -2.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 242 | 101 | 141 | 41.74% | 41.67% | 41.74% | 8.26 pp | -40 | 15 | -2.67 |

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
