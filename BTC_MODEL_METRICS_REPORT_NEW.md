# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T15:30:13.388202+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 325 | 265 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 360 | 300 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 14:00:00+00:00 | 538 | 288 | 250 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 14:00:00+00:00 | 538 | 288 | 250 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 253 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-02 12:00:00+00:00 | 253 | 93 | 160 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 288 | 142 | 146 | 49.31% | 49.17% | 49.31% | 0.69 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 288 | 140 | 148 | 48.61% | 50.42% | 48.61% | 1.39 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 288 | 138 | 150 | 47.92% | 47.92% | 47.92% | 2.08 pp | -12 | 24 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| Consolidated Hourly | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 253 | 119 | 134 | 47.04% | 46.67% | 47.04% | 2.96 pp | -15 | 16 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 265 | 127 | 138 | 47.92% | 47.08% | 47.92% | 2.08 pp | -11 | 11 | -1.00 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| Consolidated Market Hours | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 253 | 117 | 136 | 46.25% | 45.00% | 46.25% | 3.75 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 253 | 115 | 138 | 45.45% | 45.00% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 253 | 115 | 138 | 45.45% | 44.58% | 45.45% | 4.55 pp | -23 | 16 | -1.44 |
| BTC Market Hours Daily | xgb | XGBoost | 288 | 125 | 163 | 43.40% | 44.17% | 43.40% | 6.60 pp | -38 | 24 | -1.58 |
| BTC Market Hours Daily | rf | RandomForest | 288 | 124 | 164 | 43.06% | 43.33% | 43.06% | 6.94 pp | -40 | 24 | -1.67 |
| BTC Daily | nn | NN | 290 | 134 | 156 | 46.21% | 46.25% | 46.21% | 3.79 pp | -22 | 13 | -1.69 |
| BTC Hourly | transformer | Transformer | 265 | 123 | 142 | 46.42% | 47.08% | 46.42% | 3.58 pp | -19 | 11 | -1.73 |
| Consolidated Market Hours | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| Consolidated Market Hours Daily | rf | RandomForest | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 288 | 119 | 169 | 41.32% | 44.17% | 41.32% | 8.68 pp | -50 | 24 | -2.08 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 44.17% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
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
| BTC Hourly | nn | NN | 265 | 110 | 155 | 41.51% | 40.42% | 41.51% | 8.49 pp | -45 | 11 | -4.09 |
| BTC Daily | transformer | Transformer | 290 | 116 | 174 | 40.00% | 37.50% | 40.00% | 10.00 pp | -58 | 13 | -4.46 |
| BTC Hourly | rf | RandomForest | 265 | 107 | 158 | 40.38% | 40.83% | 40.38% | 9.62 pp | -51 | 11 | -4.64 |
| BTC Daily | rf | RandomForest | 290 | 110 | 180 | 37.93% | 37.08% | 37.93% | 12.07 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 300 | 112 | 188 | 37.33% | 37.92% | 37.33% | 12.67 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 290 | 104 | 186 | 35.86% | 36.25% | 35.86% | 14.14 pp | -82 | 13 | -6.31 |
| BTC Hourly | lstm | LSTM | 265 | 95 | 170 | 35.85% | 34.17% | 35.85% | 14.15 pp | -75 | 11 | -6.82 |
| BTC Hourly | xgb | XGBoost | 265 | 92 | 173 | 34.72% | 34.58% | 34.72% | 15.28 pp | -81 | 11 | -7.36 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 265 | 127 | 138 | 47.92% | 47.08% | 47.92% | 2.08 pp | -11 | 11 | -1.00 |
| BTC Hourly | transformer | Transformer | 265 | 123 | 142 | 46.42% | 47.08% | 46.42% | 3.58 pp | -19 | 11 | -1.73 |
| BTC Hourly | nn | NN | 265 | 110 | 155 | 41.51% | 40.42% | 41.51% | 8.49 pp | -45 | 11 | -4.09 |
| BTC Hourly | rf | RandomForest | 265 | 107 | 158 | 40.38% | 40.83% | 40.38% | 9.62 pp | -51 | 11 | -4.64 |
| BTC Hourly | lstm | LSTM | 265 | 95 | 170 | 35.85% | 34.17% | 35.85% | 14.15 pp | -75 | 11 | -6.82 |
| BTC Hourly | xgb | XGBoost | 265 | 92 | 173 | 34.72% | 34.58% | 34.72% | 15.28 pp | -81 | 11 | -7.36 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 290 | 134 | 156 | 46.21% | 46.25% | 46.21% | 3.79 pp | -22 | 13 | -1.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 290 | 130 | 160 | 44.83% | 44.17% | 44.83% | 5.17 pp | -30 | 13 | -2.31 |
| BTC Daily | transformer | Transformer | 290 | 116 | 174 | 40.00% | 37.50% | 40.00% | 10.00 pp | -58 | 13 | -4.46 |
| BTC Daily | rf | RandomForest | 290 | 110 | 180 | 37.93% | 37.08% | 37.93% | 12.07 pp | -70 | 13 | -5.38 |
| BTC Daily | xgb | XGBoost | 300 | 112 | 188 | 37.33% | 37.92% | 37.33% | 12.67 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 290 | 104 | 186 | 35.86% | 36.25% | 35.86% | 14.14 pp | -82 | 13 | -6.31 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 288 | 148 | 140 | 51.39% | 50.42% | 51.39% | 1.39 pp | 8 | 23 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 288 | 134 | 154 | 46.53% | 47.08% | 46.53% | 3.47 pp | -20 | 23 | -0.87 |
| BTC Market Hours | transformer | Transformer | 288 | 132 | 156 | 45.83% | 46.25% | 45.83% | 4.17 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 288 | 130 | 158 | 45.14% | 46.67% | 45.14% | 4.86 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 288 | 128 | 160 | 44.44% | 43.33% | 44.44% | 5.56 pp | -32 | 23 | -1.39 |
| BTC Market Hours | lstm | LSTM | 288 | 120 | 168 | 41.67% | 43.33% | 41.67% | 8.33 pp | -48 | 23 | -2.09 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 288 | 142 | 146 | 49.31% | 49.17% | 49.31% | 0.69 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 288 | 140 | 148 | 48.61% | 50.42% | 48.61% | 1.39 pp | -8 | 24 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 288 | 138 | 150 | 47.92% | 47.92% | 47.92% | 2.08 pp | -12 | 24 | -0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 288 | 125 | 163 | 43.40% | 44.17% | 43.40% | 6.60 pp | -38 | 24 | -1.58 |
| BTC Market Hours Daily | rf | RandomForest | 288 | 124 | 164 | 43.06% | 43.33% | 43.06% | 6.94 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 288 | 119 | 169 | 41.32% | 44.17% | 41.32% | 8.68 pp | -50 | 24 | -2.08 |

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
