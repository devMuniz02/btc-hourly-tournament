# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T22:10:47.859984+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 329 | 269 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 365 | 305 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 21:00:00+00:00 | 550 | 293 | 257 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 21:00:00+00:00 | 550 | 293 | 257 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 257 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 257 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 257 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 258 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 293 | 150 | 143 | 51.19% | 50.42% | 51.19% | 1.19 pp | 7 | 23 | 0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 293 | 144 | 149 | 49.15% | 48.75% | 49.15% | 0.85 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 293 | 143 | 150 | 48.81% | 50.42% | 48.81% | 1.19 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 293 | 140 | 153 | 47.78% | 47.92% | 47.78% | 2.22 pp | -13 | 24 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 293 | 135 | 158 | 46.08% | 46.25% | 46.08% | 3.92 pp | -23 | 23 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 269 | 128 | 141 | 47.58% | 46.25% | 47.58% | 2.42 pp | -13 | 12 | -1.08 |
| BTC Market Hours | transformer | Transformer | 293 | 134 | 159 | 45.73% | 45.83% | 45.73% | 4.27 pp | -25 | 23 | -1.09 |
| Consolidated Hourly | rf | RandomForest | 257 | 119 | 138 | 46.30% | 46.67% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 119 | 138 | 46.30% | 46.67% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 293 | 132 | 161 | 45.05% | 46.67% | 45.05% | 4.95 pp | -29 | 23 | -1.26 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.42% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.42% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Hourly | transformer | Transformer | 269 | 126 | 143 | 46.84% | 47.08% | 46.84% | 3.16 pp | -17 | 12 | -1.42 |
| BTC Market Hours | rf | RandomForest | 293 | 130 | 163 | 44.37% | 43.75% | 44.37% | 5.63 pp | -33 | 23 | -1.43 |
| Consolidated Hourly | lstm | LSTM | 257 | 117 | 140 | 45.53% | 44.58% | 45.53% | 4.47 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 117 | 140 | 45.53% | 44.58% | 45.53% | 4.47 pp | -23 | 16 | -1.44 |
| Consolidated Market Hours Daily | transformer | Transformer | 96 | 42 | 54 | 43.75% | 43.75% | 43.75% | 6.25 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 295 | 137 | 158 | 46.44% | 46.25% | 46.44% | 3.56 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | rf | RandomForest | 293 | 127 | 166 | 43.34% | 43.75% | 43.34% | 6.66 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 293 | 127 | 166 | 43.34% | 44.17% | 43.34% | 6.66 pp | -39 | 24 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 257 | 114 | 143 | 44.36% | 43.75% | 44.36% | 5.64 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 114 | 143 | 44.36% | 43.75% | 44.36% | 5.64 pp | -29 | 16 | -1.81 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 96 | 40 | 56 | 41.67% | 41.67% | 41.67% | 8.33 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 293 | 121 | 172 | 41.30% | 43.75% | 41.30% | 8.70 pp | -51 | 24 | -2.12 |
| BTC Market Hours | lstm | LSTM | 293 | 121 | 172 | 41.30% | 42.92% | 41.30% | 8.70 pp | -51 | 23 | -2.22 |
| Consolidated Hourly | nn | NN | 257 | 110 | 147 | 42.80% | 43.33% | 42.80% | 7.20 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 110 | 147 | 42.80% | 43.33% | 42.80% | 7.20 pp | -37 | 16 | -2.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 295 | 131 | 164 | 44.41% | 42.92% | 44.41% | 5.59 pp | -33 | 13 | -2.54 |
| Consolidated Hourly | xgb | XGBoost | 257 | 108 | 149 | 42.02% | 41.67% | 42.02% | 7.98 pp | -41 | 16 | -2.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 108 | 149 | 42.02% | 41.67% | 42.02% | 7.98 pp | -41 | 16 | -2.56 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 96 | 35 | 61 | 36.46% | 36.46% | 36.46% | 13.54 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 269 | 112 | 157 | 41.64% | 40.42% | 41.64% | 8.36 pp | -45 | 12 | -3.75 |
| BTC Daily | transformer | Transformer | 295 | 120 | 175 | 40.68% | 38.75% | 40.68% | 9.32 pp | -55 | 13 | -4.23 |
| BTC Hourly | rf | RandomForest | 269 | 107 | 162 | 39.78% | 40.42% | 39.78% | 10.22 pp | -55 | 12 | -4.58 |
| BTC Daily | rf | RandomForest | 295 | 114 | 181 | 38.64% | 37.92% | 38.64% | 11.36 pp | -67 | 13 | -5.15 |
| BTC Daily | xgb | XGBoost | 305 | 114 | 191 | 37.38% | 37.92% | 37.38% | 12.62 pp | -77 | 14 | -5.50 |
| BTC Hourly | lstm | LSTM | 269 | 96 | 173 | 35.69% | 34.17% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |
| BTC Daily | lstm | LSTM | 295 | 105 | 190 | 35.59% | 36.25% | 35.59% | 14.41 pp | -85 | 13 | -6.54 |
| BTC Hourly | xgb | XGBoost | 269 | 94 | 175 | 34.94% | 35.00% | 34.94% | 15.06 pp | -81 | 12 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 269 | 128 | 141 | 47.58% | 46.25% | 47.58% | 2.42 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 269 | 126 | 143 | 46.84% | 47.08% | 46.84% | 3.16 pp | -17 | 12 | -1.42 |
| BTC Hourly | nn | NN | 269 | 112 | 157 | 41.64% | 40.42% | 41.64% | 8.36 pp | -45 | 12 | -3.75 |
| BTC Hourly | rf | RandomForest | 269 | 107 | 162 | 39.78% | 40.42% | 39.78% | 10.22 pp | -55 | 12 | -4.58 |
| BTC Hourly | lstm | LSTM | 269 | 96 | 173 | 35.69% | 34.17% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 269 | 94 | 175 | 34.94% | 35.00% | 34.94% | 15.06 pp | -81 | 12 | -6.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 295 | 137 | 158 | 46.44% | 46.25% | 46.44% | 3.56 pp | -21 | 13 | -1.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 295 | 131 | 164 | 44.41% | 42.92% | 44.41% | 5.59 pp | -33 | 13 | -2.54 |
| BTC Daily | transformer | Transformer | 295 | 120 | 175 | 40.68% | 38.75% | 40.68% | 9.32 pp | -55 | 13 | -4.23 |
| BTC Daily | rf | RandomForest | 295 | 114 | 181 | 38.64% | 37.92% | 38.64% | 11.36 pp | -67 | 13 | -5.15 |
| BTC Daily | xgb | XGBoost | 305 | 114 | 191 | 37.38% | 37.92% | 37.38% | 12.62 pp | -77 | 14 | -5.50 |
| BTC Daily | lstm | LSTM | 295 | 105 | 190 | 35.59% | 36.25% | 35.59% | 14.41 pp | -85 | 13 | -6.54 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 293 | 150 | 143 | 51.19% | 50.42% | 51.19% | 1.19 pp | 7 | 23 | 0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 293 | 135 | 158 | 46.08% | 46.25% | 46.08% | 3.92 pp | -23 | 23 | -1.00 |
| BTC Market Hours | transformer | Transformer | 293 | 134 | 159 | 45.73% | 45.83% | 45.73% | 4.27 pp | -25 | 23 | -1.09 |
| BTC Market Hours | xgb | XGBoost | 293 | 132 | 161 | 45.05% | 46.67% | 45.05% | 4.95 pp | -29 | 23 | -1.26 |
| BTC Market Hours | rf | RandomForest | 293 | 130 | 163 | 44.37% | 43.75% | 44.37% | 5.63 pp | -33 | 23 | -1.43 |
| BTC Market Hours | lstm | LSTM | 293 | 121 | 172 | 41.30% | 42.92% | 41.30% | 8.70 pp | -51 | 23 | -2.22 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 293 | 144 | 149 | 49.15% | 48.75% | 49.15% | 0.85 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 293 | 143 | 150 | 48.81% | 50.42% | 48.81% | 1.19 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 293 | 140 | 153 | 47.78% | 47.92% | 47.78% | 2.22 pp | -13 | 24 | -0.54 |
| BTC Market Hours Daily | rf | RandomForest | 293 | 127 | 166 | 43.34% | 43.75% | 43.34% | 6.66 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 293 | 127 | 166 | 43.34% | 44.17% | 43.34% | 6.66 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | lstm | LSTM | 293 | 121 | 172 | 41.30% | 43.75% | 41.30% | 8.70 pp | -51 | 24 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 257 | 119 | 138 | 46.30% | 46.67% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.42% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | lstm | LSTM | 257 | 117 | 140 | 45.53% | 44.58% | 45.53% | 4.47 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 257 | 114 | 143 | 44.36% | 43.75% | 44.36% | 5.64 pp | -29 | 16 | -1.81 |
| Consolidated Hourly | nn | NN | 257 | 110 | 147 | 42.80% | 43.33% | 42.80% | 7.20 pp | -37 | 16 | -2.31 |
| Consolidated Hourly | xgb | XGBoost | 257 | 108 | 149 | 42.02% | 41.67% | 42.02% | 7.98 pp | -41 | 16 | -2.56 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 119 | 138 | 46.30% | 46.67% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.42% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 117 | 140 | 45.53% | 44.58% | 45.53% | 4.47 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 114 | 143 | 44.36% | 43.75% | 44.36% | 5.64 pp | -29 | 16 | -1.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 110 | 147 | 42.80% | 43.33% | 42.80% | 7.20 pp | -37 | 16 | -2.31 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 108 | 149 | 42.02% | 41.67% | 42.02% | 7.98 pp | -41 | 16 | -2.56 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 96 | 42 | 54 | 43.75% | 43.75% | 43.75% | 6.25 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 96 | 40 | 56 | 41.67% | 41.67% | 41.67% | 8.33 pp | -16 | 8 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 96 | 35 | 61 | 36.46% | 36.46% | 36.46% | 13.54 pp | -26 | 8 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
