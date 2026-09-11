# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T17:00:20.863207+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 326 | 266 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 361 | 301 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 540 | 289 | 251 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 15:00:00+00:00 | 540 | 289 | 251 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 289 | 148 | 141 | 51.21% | 50.42% | 51.21% | 1.21 pp | 7 | 23 | 0.30 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 289 | 142 | 147 | 49.13% | 49.17% | 49.13% | 0.87 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 289 | 141 | 148 | 48.79% | 50.42% | 48.79% | 1.21 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 289 | 138 | 151 | 47.75% | 47.92% | 47.75% | 2.25 pp | -13 | 24 | -0.54 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 266 | 128 | 138 | 48.12% | 47.08% | 48.12% | 1.88 pp | -10 | 12 | -0.83 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 289 | 134 | 155 | 46.37% | 46.67% | 46.37% | 3.63 pp | -21 | 23 | -0.91 |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| BTC Market Hours | transformer | Transformer | 289 | 133 | 156 | 46.02% | 46.25% | 46.02% | 3.98 pp | -23 | 23 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 289 | 131 | 158 | 45.33% | 46.67% | 45.33% | 4.67 pp | -27 | 23 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| BTC Market Hours | rf | RandomForest | 289 | 128 | 161 | 44.29% | 43.33% | 44.29% | 5.71 pp | -33 | 23 | -1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| BTC Hourly | transformer | Transformer | 266 | 124 | 142 | 46.62% | 47.08% | 46.62% | 3.38 pp | -18 | 12 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 289 | 125 | 164 | 43.25% | 43.33% | 43.25% | 6.75 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 289 | 125 | 164 | 43.25% | 43.75% | 43.25% | 6.75 pp | -39 | 24 | -1.62 |
| BTC Daily | nn | NN | 291 | 134 | 157 | 46.05% | 45.83% | 46.05% | 3.95 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 289 | 120 | 169 | 41.52% | 44.17% | 41.52% | 8.48 pp | -49 | 24 | -2.04 |
| BTC Market Hours | lstm | LSTM | 289 | 120 | 169 | 41.52% | 43.33% | 41.52% | 8.48 pp | -49 | 23 | -2.13 |
| BTC Daily | mlp_sklearn | MLPClassifier | 291 | 130 | 161 | 44.67% | 43.75% | 44.67% | 5.33 pp | -31 | 13 | -2.38 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 266 | 111 | 155 | 41.73% | 40.83% | 41.73% | 8.27 pp | -44 | 12 | -3.67 |
| BTC Hourly | rf | RandomForest | 266 | 107 | 159 | 40.23% | 40.42% | 40.23% | 9.77 pp | -52 | 12 | -4.33 |
| BTC Daily | transformer | Transformer | 291 | 117 | 174 | 40.21% | 37.92% | 40.21% | 9.79 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 291 | 111 | 180 | 38.14% | 37.50% | 38.14% | 11.86 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 301 | 112 | 189 | 37.21% | 37.92% | 37.21% | 12.79 pp | -77 | 14 | -5.50 |
| BTC Hourly | lstm | LSTM | 266 | 96 | 170 | 36.09% | 34.58% | 36.09% | 13.91 pp | -74 | 12 | -6.17 |
| BTC Daily | lstm | LSTM | 291 | 104 | 187 | 35.74% | 35.83% | 35.74% | 14.26 pp | -83 | 13 | -6.38 |
| BTC Hourly | xgb | XGBoost | 266 | 93 | 173 | 34.96% | 34.58% | 34.96% | 15.04 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 266 | 128 | 138 | 48.12% | 47.08% | 48.12% | 1.88 pp | -10 | 12 | -0.83 |
| BTC Hourly | transformer | Transformer | 266 | 124 | 142 | 46.62% | 47.08% | 46.62% | 3.38 pp | -18 | 12 | -1.50 |
| BTC Hourly | nn | NN | 266 | 111 | 155 | 41.73% | 40.83% | 41.73% | 8.27 pp | -44 | 12 | -3.67 |
| BTC Hourly | rf | RandomForest | 266 | 107 | 159 | 40.23% | 40.42% | 40.23% | 9.77 pp | -52 | 12 | -4.33 |
| BTC Hourly | lstm | LSTM | 266 | 96 | 170 | 36.09% | 34.58% | 36.09% | 13.91 pp | -74 | 12 | -6.17 |
| BTC Hourly | xgb | XGBoost | 266 | 93 | 173 | 34.96% | 34.58% | 34.96% | 15.04 pp | -80 | 12 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 291 | 134 | 157 | 46.05% | 45.83% | 46.05% | 3.95 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 291 | 130 | 161 | 44.67% | 43.75% | 44.67% | 5.33 pp | -31 | 13 | -2.38 |
| BTC Daily | transformer | Transformer | 291 | 117 | 174 | 40.21% | 37.92% | 40.21% | 9.79 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 291 | 111 | 180 | 38.14% | 37.50% | 38.14% | 11.86 pp | -69 | 13 | -5.31 |
| BTC Daily | xgb | XGBoost | 301 | 112 | 189 | 37.21% | 37.92% | 37.21% | 12.79 pp | -77 | 14 | -5.50 |
| BTC Daily | lstm | LSTM | 291 | 104 | 187 | 35.74% | 35.83% | 35.74% | 14.26 pp | -83 | 13 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 289 | 148 | 141 | 51.21% | 50.42% | 51.21% | 1.21 pp | 7 | 23 | 0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 289 | 134 | 155 | 46.37% | 46.67% | 46.37% | 3.63 pp | -21 | 23 | -0.91 |
| BTC Market Hours | transformer | Transformer | 289 | 133 | 156 | 46.02% | 46.25% | 46.02% | 3.98 pp | -23 | 23 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 289 | 131 | 158 | 45.33% | 46.67% | 45.33% | 4.67 pp | -27 | 23 | -1.17 |
| BTC Market Hours | rf | RandomForest | 289 | 128 | 161 | 44.29% | 43.33% | 44.29% | 5.71 pp | -33 | 23 | -1.43 |
| BTC Market Hours | lstm | LSTM | 289 | 120 | 169 | 41.52% | 43.33% | 41.52% | 8.48 pp | -49 | 23 | -2.13 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 289 | 142 | 147 | 49.13% | 49.17% | 49.13% | 0.87 pp | -5 | 24 | -0.21 |
| BTC Market Hours Daily | nn | NN | 289 | 141 | 148 | 48.79% | 50.42% | 48.79% | 1.21 pp | -7 | 24 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 289 | 138 | 151 | 47.75% | 47.92% | 47.75% | 2.25 pp | -13 | 24 | -0.54 |
| BTC Market Hours Daily | rf | RandomForest | 289 | 125 | 164 | 43.25% | 43.33% | 43.25% | 6.75 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 289 | 125 | 164 | 43.25% | 43.75% | 43.25% | 6.75 pp | -39 | 24 | -1.62 |
| BTC Market Hours Daily | lstm | LSTM | 289 | 120 | 169 | 41.52% | 44.17% | 41.52% | 8.48 pp | -49 | 24 | -2.04 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Hourly | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 253 | 104 | 149 | 41.11% | 42.08% | 41.11% | 8.89 pp | -45 | 16 | -2.81 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 253 | 103 | 150 | 40.71% | 40.83% | 40.71% | 9.29 pp | -47 | 16 | -2.94 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | nn | NN | 93 | 35 | 58 | 37.63% | 37.63% | 37.63% | 12.37 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 93 | 34 | 59 | 36.56% | 36.56% | 36.56% | 13.44 pp | -25 | 8 | -3.12 |
| Consolidated Market Hours Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 8 | -3.38 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
