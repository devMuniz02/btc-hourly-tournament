# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T23:44:13.176113+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 330 | 270 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 366 | 306 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 22:00:00+00:00 | 552 | 294 | 258 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 22:00:00+00:00 | 552 | 294 | 258 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 258 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 258 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 258 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T14:00:00+00:00 | 259 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 294 | 150 | 144 | 51.02% | 50.00% | 51.02% | 1.02 pp | 6 | 23 | 0.26 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 294 | 144 | 150 | 48.98% | 48.75% | 48.98% | 1.02 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | nn | NN | 294 | 143 | 151 | 48.64% | 50.42% | 48.64% | 1.36 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 294 | 140 | 154 | 47.62% | 47.50% | 47.62% | 2.38 pp | -14 | 24 | -0.58 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 294 | 136 | 158 | 46.26% | 46.67% | 46.26% | 3.74 pp | -22 | 23 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 270 | 129 | 141 | 47.78% | 46.67% | 47.78% | 2.22 pp | -12 | 12 | -1.00 |
| BTC Market Hours | transformer | Transformer | 294 | 135 | 159 | 45.92% | 46.25% | 45.92% | 4.08 pp | -24 | 23 | -1.04 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.42% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Hourly | rf | RandomForest | 258 | 119 | 139 | 46.12% | 46.25% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.42% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 258 | 119 | 139 | 46.12% | 46.25% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 294 | 132 | 162 | 44.90% | 46.67% | 44.90% | 5.10 pp | -30 | 23 | -1.30 |
| BTC Hourly | transformer | Transformer | 270 | 127 | 143 | 47.04% | 47.50% | 47.04% | 2.96 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 258 | 118 | 140 | 45.74% | 44.58% | 45.74% | 4.26 pp | -22 | 16 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 258 | 118 | 140 | 45.74% | 44.58% | 45.74% | 4.26 pp | -22 | 16 | -1.38 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | rf | RandomForest | 294 | 130 | 164 | 44.22% | 43.75% | 44.22% | 5.78 pp | -34 | 23 | -1.48 |
| Consolidated Market Hours Daily | transformer | Transformer | 96 | 42 | 54 | 43.75% | 43.75% | 43.75% | 6.25 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 294 | 127 | 167 | 43.20% | 43.75% | 43.20% | 6.80 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 294 | 127 | 167 | 43.20% | 44.17% | 43.20% | 6.80 pp | -40 | 24 | -1.67 |
| BTC Daily | nn | NN | 296 | 137 | 159 | 46.28% | 46.25% | 46.28% | 3.72 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | transformer | Transformer | 258 | 114 | 144 | 44.19% | 43.33% | 44.19% | 5.81 pp | -30 | 16 | -1.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 258 | 114 | 144 | 44.19% | 43.33% | 44.19% | 5.81 pp | -30 | 16 | -1.88 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 96 | 40 | 56 | 41.67% | 41.67% | 41.67% | 8.33 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 294 | 121 | 173 | 41.16% | 43.75% | 41.16% | 8.84 pp | -52 | 24 | -2.17 |
| BTC Market Hours | lstm | LSTM | 294 | 122 | 172 | 41.50% | 42.92% | 41.50% | 8.50 pp | -50 | 23 | -2.17 |
| Consolidated Hourly | nn | NN | 258 | 111 | 147 | 43.02% | 43.33% | 43.02% | 6.98 pp | -36 | 16 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 258 | 111 | 147 | 43.02% | 43.33% | 43.02% | 6.98 pp | -36 | 16 | -2.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 296 | 131 | 165 | 44.26% | 42.92% | 44.26% | 5.74 pp | -34 | 13 | -2.62 |
| Consolidated Hourly | xgb | XGBoost | 258 | 108 | 150 | 41.86% | 41.25% | 41.86% | 8.14 pp | -42 | 16 | -2.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 258 | 108 | 150 | 41.86% | 41.25% | 41.86% | 8.14 pp | -42 | 16 | -2.62 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 96 | 35 | 61 | 36.46% | 36.46% | 36.46% | 13.54 pp | -26 | 8 | -3.25 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 270 | 113 | 157 | 41.85% | 40.83% | 41.85% | 8.15 pp | -44 | 12 | -3.67 |
| BTC Daily | transformer | Transformer | 296 | 120 | 176 | 40.54% | 38.33% | 40.54% | 9.46 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 270 | 108 | 162 | 40.00% | 40.83% | 40.00% | 10.00 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 296 | 114 | 182 | 38.51% | 37.92% | 38.51% | 11.49 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 306 | 114 | 192 | 37.25% | 37.92% | 37.25% | 12.75 pp | -78 | 14 | -5.57 |
| BTC Daily | lstm | LSTM | 296 | 106 | 190 | 35.81% | 36.25% | 35.81% | 14.19 pp | -84 | 13 | -6.46 |
| BTC Hourly | lstm | LSTM | 270 | 96 | 174 | 35.56% | 34.17% | 35.56% | 14.44 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 270 | 95 | 175 | 35.19% | 35.42% | 35.19% | 14.81 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 270 | 129 | 141 | 47.78% | 46.67% | 47.78% | 2.22 pp | -12 | 12 | -1.00 |
| BTC Hourly | transformer | Transformer | 270 | 127 | 143 | 47.04% | 47.50% | 47.04% | 2.96 pp | -16 | 12 | -1.33 |
| BTC Hourly | nn | NN | 270 | 113 | 157 | 41.85% | 40.83% | 41.85% | 8.15 pp | -44 | 12 | -3.67 |
| BTC Hourly | rf | RandomForest | 270 | 108 | 162 | 40.00% | 40.83% | 40.00% | 10.00 pp | -54 | 12 | -4.50 |
| BTC Hourly | lstm | LSTM | 270 | 96 | 174 | 35.56% | 34.17% | 35.56% | 14.44 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 270 | 95 | 175 | 35.19% | 35.42% | 35.19% | 14.81 pp | -80 | 12 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 296 | 137 | 159 | 46.28% | 46.25% | 46.28% | 3.72 pp | -22 | 13 | -1.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 296 | 131 | 165 | 44.26% | 42.92% | 44.26% | 5.74 pp | -34 | 13 | -2.62 |
| BTC Daily | transformer | Transformer | 296 | 120 | 176 | 40.54% | 38.33% | 40.54% | 9.46 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 296 | 114 | 182 | 38.51% | 37.92% | 38.51% | 11.49 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 306 | 114 | 192 | 37.25% | 37.92% | 37.25% | 12.75 pp | -78 | 14 | -5.57 |
| BTC Daily | lstm | LSTM | 296 | 106 | 190 | 35.81% | 36.25% | 35.81% | 14.19 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 294 | 150 | 144 | 51.02% | 50.00% | 51.02% | 1.02 pp | 6 | 23 | 0.26 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 294 | 136 | 158 | 46.26% | 46.67% | 46.26% | 3.74 pp | -22 | 23 | -0.96 |
| BTC Market Hours | transformer | Transformer | 294 | 135 | 159 | 45.92% | 46.25% | 45.92% | 4.08 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 294 | 132 | 162 | 44.90% | 46.67% | 44.90% | 5.10 pp | -30 | 23 | -1.30 |
| BTC Market Hours | rf | RandomForest | 294 | 130 | 164 | 44.22% | 43.75% | 44.22% | 5.78 pp | -34 | 23 | -1.48 |
| BTC Market Hours | lstm | LSTM | 294 | 122 | 172 | 41.50% | 42.92% | 41.50% | 8.50 pp | -50 | 23 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 294 | 144 | 150 | 48.98% | 48.75% | 48.98% | 1.02 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | nn | NN | 294 | 143 | 151 | 48.64% | 50.42% | 48.64% | 1.36 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 294 | 140 | 154 | 47.62% | 47.50% | 47.62% | 2.38 pp | -14 | 24 | -0.58 |
| BTC Market Hours Daily | rf | RandomForest | 294 | 127 | 167 | 43.20% | 43.75% | 43.20% | 6.80 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 294 | 127 | 167 | 43.20% | 44.17% | 43.20% | 6.80 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 294 | 121 | 173 | 41.16% | 43.75% | 41.16% | 8.84 pp | -52 | 24 | -2.17 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.42% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Hourly | rf | RandomForest | 258 | 119 | 139 | 46.12% | 46.25% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 258 | 118 | 140 | 45.74% | 44.58% | 45.74% | 4.26 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 258 | 114 | 144 | 44.19% | 43.33% | 44.19% | 5.81 pp | -30 | 16 | -1.88 |
| Consolidated Hourly | nn | NN | 258 | 111 | 147 | 43.02% | 43.33% | 43.02% | 6.98 pp | -36 | 16 | -2.25 |
| Consolidated Hourly | xgb | XGBoost | 258 | 108 | 150 | 41.86% | 41.25% | 41.86% | 8.14 pp | -42 | 16 | -2.62 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 258 | 119 | 139 | 46.12% | 45.42% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 258 | 119 | 139 | 46.12% | 46.25% | 46.12% | 3.88 pp | -20 | 16 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 258 | 118 | 140 | 45.74% | 44.58% | 45.74% | 4.26 pp | -22 | 16 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 258 | 114 | 144 | 44.19% | 43.33% | 44.19% | 5.81 pp | -30 | 16 | -1.88 |
| Consolidated Daily/Hourly Refresh | nn | NN | 258 | 111 | 147 | 43.02% | 43.33% | 43.02% | 6.98 pp | -36 | 16 | -2.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 258 | 108 | 150 | 41.86% | 41.25% | 41.86% | 8.14 pp | -42 | 16 | -2.62 |

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
