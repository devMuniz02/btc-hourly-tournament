# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T21:41:30.227428+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 364 | 304 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 20:00:00+00:00 | 548 | 292 | 256 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 20:00:00+00:00 | 548 | 292 | 256 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 257 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 14:00:00+00:00 | 257 | 95 | 162 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 292 | 150 | 142 | 51.37% | 50.42% | 51.37% | 1.37 pp | 8 | 23 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 292 | 144 | 148 | 49.32% | 49.17% | 49.32% | 0.68 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 292 | 143 | 149 | 48.97% | 50.83% | 48.97% | 1.03 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 292 | 140 | 152 | 47.95% | 47.92% | 47.95% | 2.05 pp | -12 | 24 | -0.50 |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 292 | 135 | 157 | 46.23% | 46.25% | 46.23% | 3.77 pp | -22 | 23 | -0.96 |
| BTC Market Hours | transformer | Transformer | 292 | 134 | 158 | 45.89% | 46.25% | 45.89% | 4.11 pp | -24 | 23 | -1.04 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 269 | 128 | 141 | 47.58% | 46.25% | 47.58% | 2.42 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 292 | 132 | 160 | 45.21% | 46.67% | 45.21% | 4.79 pp | -28 | 23 | -1.22 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Market Hours | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | rf | RandomForest | 292 | 130 | 162 | 44.52% | 43.75% | 44.52% | 5.48 pp | -32 | 23 | -1.39 |
| BTC Hourly | transformer | Transformer | 269 | 126 | 143 | 46.84% | 47.08% | 46.84% | 3.16 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| BTC Market Hours Daily | rf | RandomForest | 292 | 126 | 166 | 43.15% | 43.33% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 292 | 126 | 166 | 43.15% | 43.75% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Daily | nn | NN | 294 | 136 | 158 | 46.26% | 46.25% | 46.26% | 3.74 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 292 | 121 | 171 | 41.44% | 44.17% | 41.44% | 8.56 pp | -50 | 24 | -2.08 |
| BTC Market Hours | lstm | LSTM | 292 | 121 | 171 | 41.44% | 42.92% | 41.44% | 8.56 pp | -50 | 23 | -2.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 294 | 130 | 164 | 44.22% | 42.92% | 44.22% | 5.78 pp | -34 | 13 | -2.62 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |
| Consolidated Market Hours | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 269 | 112 | 157 | 41.64% | 40.42% | 41.64% | 8.36 pp | -45 | 12 | -3.75 |
| BTC Daily | transformer | Transformer | 294 | 119 | 175 | 40.48% | 38.75% | 40.48% | 9.52 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 269 | 107 | 162 | 39.78% | 40.42% | 39.78% | 10.22 pp | -55 | 12 | -4.58 |
| BTC Daily | rf | RandomForest | 294 | 113 | 181 | 38.44% | 37.92% | 38.44% | 11.56 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 304 | 113 | 191 | 37.17% | 37.92% | 37.17% | 12.83 pp | -78 | 14 | -5.57 |
| BTC Hourly | lstm | LSTM | 269 | 96 | 173 | 35.69% | 34.17% | 35.69% | 14.31 pp | -77 | 12 | -6.42 |
| BTC Daily | lstm | LSTM | 294 | 105 | 189 | 35.71% | 36.25% | 35.71% | 14.29 pp | -84 | 13 | -6.46 |
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
| BTC Daily | nn | NN | 294 | 136 | 158 | 46.26% | 46.25% | 46.26% | 3.74 pp | -22 | 13 | -1.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 294 | 130 | 164 | 44.22% | 42.92% | 44.22% | 5.78 pp | -34 | 13 | -2.62 |
| BTC Daily | transformer | Transformer | 294 | 119 | 175 | 40.48% | 38.75% | 40.48% | 9.52 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 294 | 113 | 181 | 38.44% | 37.92% | 38.44% | 11.56 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 304 | 113 | 191 | 37.17% | 37.92% | 37.17% | 12.83 pp | -78 | 14 | -5.57 |
| BTC Daily | lstm | LSTM | 294 | 105 | 189 | 35.71% | 36.25% | 35.71% | 14.29 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 292 | 150 | 142 | 51.37% | 50.42% | 51.37% | 1.37 pp | 8 | 23 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 292 | 135 | 157 | 46.23% | 46.25% | 46.23% | 3.77 pp | -22 | 23 | -0.96 |
| BTC Market Hours | transformer | Transformer | 292 | 134 | 158 | 45.89% | 46.25% | 45.89% | 4.11 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 292 | 132 | 160 | 45.21% | 46.67% | 45.21% | 4.79 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 292 | 130 | 162 | 44.52% | 43.75% | 44.52% | 5.48 pp | -32 | 23 | -1.39 |
| BTC Market Hours | lstm | LSTM | 292 | 121 | 171 | 41.44% | 42.92% | 41.44% | 8.56 pp | -50 | 23 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 292 | 144 | 148 | 49.32% | 49.17% | 49.32% | 0.68 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 292 | 143 | 149 | 48.97% | 50.83% | 48.97% | 1.03 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 292 | 140 | 152 | 47.95% | 47.92% | 47.95% | 2.05 pp | -12 | 24 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 292 | 126 | 166 | 43.15% | 43.33% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 292 | 126 | 166 | 43.15% | 43.75% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 292 | 121 | 171 | 41.44% | 44.17% | 41.44% | 8.56 pp | -50 | 24 | -2.08 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Hourly | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 257 | 118 | 139 | 45.91% | 45.00% | 45.91% | 4.09 pp | -21 | 16 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 257 | 116 | 141 | 45.14% | 44.17% | 45.14% | 4.86 pp | -25 | 16 | -1.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 257 | 106 | 151 | 41.25% | 42.08% | 41.25% | 8.75 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 257 | 104 | 153 | 40.47% | 40.42% | 40.47% | 9.53 pp | -49 | 16 | -3.06 |

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
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
