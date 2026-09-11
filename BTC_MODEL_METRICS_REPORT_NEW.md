# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T20:06:24.653916+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 328 | 268 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 363 | 303 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 546 | 291 | 255 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 546 | 291 | 255 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 255 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 13:00:00+00:00 | 255 | 94 | 161 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 291 | 149 | 142 | 51.20% | 50.42% | 51.20% | 1.20 pp | 7 | 23 | 0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 291 | 143 | 148 | 49.14% | 49.17% | 49.14% | 0.86 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 291 | 142 | 149 | 48.80% | 50.42% | 48.80% | 1.20 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 291 | 139 | 152 | 47.77% | 47.92% | 47.77% | 2.23 pp | -13 | 24 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 291 | 135 | 156 | 46.39% | 46.67% | 46.39% | 3.61 pp | -21 | 23 | -0.91 |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 268 | 128 | 140 | 47.76% | 46.25% | 47.76% | 2.24 pp | -12 | 12 | -1.00 |
| BTC Market Hours | transformer | Transformer | 291 | 134 | 157 | 46.05% | 46.25% | 46.05% | 3.95 pp | -23 | 23 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 291 | 131 | 160 | 45.02% | 46.25% | 45.02% | 4.98 pp | -29 | 23 | -1.26 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| BTC Market Hours | rf | RandomForest | 291 | 129 | 162 | 44.33% | 43.33% | 44.33% | 5.67 pp | -33 | 23 | -1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| BTC Hourly | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| BTC Market Hours Daily | rf | RandomForest | 291 | 126 | 165 | 43.30% | 43.75% | 43.30% | 6.70 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 291 | 126 | 165 | 43.30% | 43.75% | 43.30% | 6.70 pp | -39 | 24 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 293 | 135 | 158 | 46.08% | 45.83% | 46.08% | 3.92 pp | -23 | 13 | -1.77 |
| BTC Market Hours Daily | lstm | LSTM | 291 | 120 | 171 | 41.24% | 43.75% | 41.24% | 8.76 pp | -51 | 24 | -2.12 |
| BTC Market Hours | lstm | LSTM | 291 | 121 | 170 | 41.58% | 43.33% | 41.58% | 8.42 pp | -49 | 23 | -2.13 |
| BTC Daily | mlp_sklearn | MLPClassifier | 293 | 130 | 163 | 44.37% | 43.33% | 44.37% | 5.63 pp | -33 | 13 | -2.54 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| BTC Hourly | nn | NN | 268 | 111 | 157 | 41.42% | 40.42% | 41.42% | 8.58 pp | -46 | 12 | -3.83 |
| BTC Daily | transformer | Transformer | 293 | 118 | 175 | 40.27% | 38.33% | 40.27% | 9.73 pp | -57 | 13 | -4.38 |
| BTC Hourly | rf | RandomForest | 268 | 107 | 161 | 39.93% | 40.42% | 39.93% | 10.07 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 293 | 112 | 181 | 38.23% | 37.50% | 38.23% | 11.77 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 303 | 113 | 190 | 37.29% | 37.92% | 37.29% | 12.71 pp | -77 | 14 | -5.50 |
| BTC Hourly | lstm | LSTM | 268 | 96 | 172 | 35.82% | 34.58% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 293 | 105 | 188 | 35.84% | 36.25% | 35.84% | 14.16 pp | -83 | 13 | -6.38 |
| BTC Hourly | xgb | XGBoost | 268 | 94 | 174 | 35.07% | 35.00% | 35.07% | 14.93 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 268 | 128 | 140 | 47.76% | 46.25% | 47.76% | 2.24 pp | -12 | 12 | -1.00 |
| BTC Hourly | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 12 | -1.50 |
| BTC Hourly | nn | NN | 268 | 111 | 157 | 41.42% | 40.42% | 41.42% | 8.58 pp | -46 | 12 | -3.83 |
| BTC Hourly | rf | RandomForest | 268 | 107 | 161 | 39.93% | 40.42% | 39.93% | 10.07 pp | -54 | 12 | -4.50 |
| BTC Hourly | lstm | LSTM | 268 | 96 | 172 | 35.82% | 34.58% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 268 | 94 | 174 | 35.07% | 35.00% | 35.07% | 14.93 pp | -80 | 12 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 293 | 135 | 158 | 46.08% | 45.83% | 46.08% | 3.92 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 293 | 130 | 163 | 44.37% | 43.33% | 44.37% | 5.63 pp | -33 | 13 | -2.54 |
| BTC Daily | transformer | Transformer | 293 | 118 | 175 | 40.27% | 38.33% | 40.27% | 9.73 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 293 | 112 | 181 | 38.23% | 37.50% | 38.23% | 11.77 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 303 | 113 | 190 | 37.29% | 37.92% | 37.29% | 12.71 pp | -77 | 14 | -5.50 |
| BTC Daily | lstm | LSTM | 293 | 105 | 188 | 35.84% | 36.25% | 35.84% | 14.16 pp | -83 | 13 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 291 | 149 | 142 | 51.20% | 50.42% | 51.20% | 1.20 pp | 7 | 23 | 0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 291 | 135 | 156 | 46.39% | 46.67% | 46.39% | 3.61 pp | -21 | 23 | -0.91 |
| BTC Market Hours | transformer | Transformer | 291 | 134 | 157 | 46.05% | 46.25% | 46.05% | 3.95 pp | -23 | 23 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 291 | 131 | 160 | 45.02% | 46.25% | 45.02% | 4.98 pp | -29 | 23 | -1.26 |
| BTC Market Hours | rf | RandomForest | 291 | 129 | 162 | 44.33% | 43.33% | 44.33% | 5.67 pp | -33 | 23 | -1.43 |
| BTC Market Hours | lstm | LSTM | 291 | 121 | 170 | 41.58% | 43.33% | 41.58% | 8.42 pp | -49 | 23 | -2.13 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 291 | 143 | 148 | 49.14% | 49.17% | 49.14% | 0.86 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 291 | 142 | 149 | 48.80% | 50.42% | 48.80% | 1.20 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 291 | 139 | 152 | 47.77% | 47.92% | 47.77% | 2.23 pp | -13 | 24 | -0.54 |
| BTC Market Hours Daily | rf | RandomForest | 291 | 126 | 165 | 43.30% | 43.75% | 43.30% | 6.70 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 291 | 126 | 165 | 43.30% | 43.75% | 43.30% | 6.70 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | lstm | LSTM | 291 | 120 | 171 | 41.24% | 43.75% | 41.24% | 8.76 pp | -51 | 24 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 255 | 120 | 135 | 47.06% | 46.67% | 47.06% | 2.94 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 255 | 117 | 138 | 45.88% | 45.00% | 45.88% | 4.12 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 255 | 116 | 139 | 45.49% | 44.58% | 45.49% | 4.51 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 255 | 115 | 140 | 45.10% | 44.17% | 45.10% | 4.90 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 255 | 105 | 150 | 41.18% | 42.08% | 41.18% | 8.82 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 255 | 103 | 152 | 40.39% | 40.00% | 40.39% | 9.61 pp | -49 | 16 | -3.06 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
