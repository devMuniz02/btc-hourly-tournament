# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T01:57:07.026414+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 316 | 256 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 351 | 291 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 526 | 279 | 247 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 00:00:00+00:00 | 526 | 279 | 247 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 245 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-06-01 20:00:00+00:00 | 245 | 88 | 157 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 279 | 142 | 137 | 50.90% | 50.42% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 279 | 137 | 142 | 49.10% | 48.33% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 279 | 134 | 145 | 48.03% | 47.92% | 48.03% | 1.97 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | nn | NN | 279 | 133 | 146 | 47.67% | 48.33% | 47.67% | 2.33 pp | -13 | 23 | -0.57 |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 256 | 123 | 133 | 48.05% | 47.92% | 48.05% | 1.95 pp | -10 | 11 | -0.91 |
| BTC Market Hours | transformer | Transformer | 279 | 129 | 150 | 46.24% | 45.83% | 46.24% | 3.76 pp | -21 | 22 | -0.95 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 279 | 128 | 151 | 45.88% | 45.83% | 45.88% | 4.12 pp | -23 | 22 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 279 | 126 | 153 | 45.16% | 45.00% | 45.16% | 4.84 pp | -27 | 22 | -1.23 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| BTC Market Hours | rf | RandomForest | 279 | 123 | 156 | 44.09% | 42.92% | 44.09% | 5.91 pp | -33 | 22 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 279 | 120 | 159 | 43.01% | 42.50% | 43.01% | 6.99 pp | -39 | 23 | -1.70 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | rf | RandomForest | 279 | 119 | 160 | 42.65% | 42.50% | 42.65% | 7.35 pp | -41 | 23 | -1.78 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| BTC Daily | nn | NN | 281 | 128 | 153 | 45.55% | 45.00% | 45.55% | 4.45 pp | -25 | 13 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 281 | 127 | 154 | 45.20% | 45.00% | 45.20% | 4.80 pp | -27 | 13 | -2.08 |
| BTC Hourly | transformer | Transformer | 256 | 116 | 140 | 45.31% | 45.83% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 279 | 114 | 165 | 40.86% | 42.92% | 40.86% | 9.14 pp | -51 | 23 | -2.22 |
| BTC Market Hours | lstm | LSTM | 279 | 115 | 164 | 41.22% | 42.50% | 41.22% | 8.78 pp | -49 | 22 | -2.23 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |
| BTC Hourly | nn | NN | 256 | 108 | 148 | 42.19% | 42.50% | 42.19% | 7.81 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 11 | -4.18 |
| BTC Daily | transformer | Transformer | 281 | 112 | 169 | 39.86% | 37.50% | 39.86% | 10.14 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 281 | 105 | 176 | 37.37% | 36.67% | 37.37% | 12.63 pp | -71 | 13 | -5.46 |
| BTC Daily | xgb | XGBoost | 291 | 107 | 184 | 36.77% | 37.08% | 36.77% | 13.23 pp | -77 | 14 | -5.50 |
| BTC Daily | lstm | LSTM | 281 | 101 | 180 | 35.94% | 36.25% | 35.94% | 14.06 pp | -79 | 13 | -6.08 |
| BTC Hourly | lstm | LSTM | 256 | 94 | 162 | 36.72% | 36.25% | 36.72% | 13.28 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 256 | 89 | 167 | 34.77% | 35.00% | 34.77% | 15.23 pp | -78 | 11 | -7.09 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 256 | 123 | 133 | 48.05% | 47.92% | 48.05% | 1.95 pp | -10 | 11 | -0.91 |
| BTC Hourly | transformer | Transformer | 256 | 116 | 140 | 45.31% | 45.83% | 45.31% | 4.69 pp | -24 | 11 | -2.18 |
| BTC Hourly | nn | NN | 256 | 108 | 148 | 42.19% | 42.50% | 42.19% | 7.81 pp | -40 | 11 | -3.64 |
| BTC Hourly | rf | RandomForest | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 11 | -4.18 |
| BTC Hourly | lstm | LSTM | 256 | 94 | 162 | 36.72% | 36.25% | 36.72% | 13.28 pp | -68 | 11 | -6.18 |
| BTC Hourly | xgb | XGBoost | 256 | 89 | 167 | 34.77% | 35.00% | 34.77% | 15.23 pp | -78 | 11 | -7.09 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 281 | 128 | 153 | 45.55% | 45.00% | 45.55% | 4.45 pp | -25 | 13 | -1.92 |
| BTC Daily | mlp_sklearn | MLPClassifier | 281 | 127 | 154 | 45.20% | 45.00% | 45.20% | 4.80 pp | -27 | 13 | -2.08 |
| BTC Daily | transformer | Transformer | 281 | 112 | 169 | 39.86% | 37.50% | 39.86% | 10.14 pp | -57 | 13 | -4.38 |
| BTC Daily | rf | RandomForest | 281 | 105 | 176 | 37.37% | 36.67% | 37.37% | 12.63 pp | -71 | 13 | -5.46 |
| BTC Daily | xgb | XGBoost | 291 | 107 | 184 | 36.77% | 37.08% | 36.77% | 13.23 pp | -77 | 14 | -5.50 |
| BTC Daily | lstm | LSTM | 281 | 101 | 180 | 35.94% | 36.25% | 35.94% | 14.06 pp | -79 | 13 | -6.08 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 279 | 142 | 137 | 50.90% | 50.42% | 50.90% | 0.90 pp | 5 | 22 | 0.23 |
| BTC Market Hours | transformer | Transformer | 279 | 129 | 150 | 46.24% | 45.83% | 46.24% | 3.76 pp | -21 | 22 | -0.95 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 279 | 128 | 151 | 45.88% | 45.83% | 45.88% | 4.12 pp | -23 | 22 | -1.05 |
| BTC Market Hours | xgb | XGBoost | 279 | 126 | 153 | 45.16% | 45.00% | 45.16% | 4.84 pp | -27 | 22 | -1.23 |
| BTC Market Hours | rf | RandomForest | 279 | 123 | 156 | 44.09% | 42.92% | 44.09% | 5.91 pp | -33 | 22 | -1.50 |
| BTC Market Hours | lstm | LSTM | 279 | 115 | 164 | 41.22% | 42.50% | 41.22% | 8.78 pp | -49 | 22 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 279 | 137 | 142 | 49.10% | 48.33% | 49.10% | 0.90 pp | -5 | 23 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 279 | 134 | 145 | 48.03% | 47.92% | 48.03% | 1.97 pp | -11 | 23 | -0.48 |
| BTC Market Hours Daily | nn | NN | 279 | 133 | 146 | 47.67% | 48.33% | 47.67% | 2.33 pp | -13 | 23 | -0.57 |
| BTC Market Hours Daily | xgb | XGBoost | 279 | 120 | 159 | 43.01% | 42.50% | 43.01% | 6.99 pp | -39 | 23 | -1.70 |
| BTC Market Hours Daily | rf | RandomForest | 279 | 119 | 160 | 42.65% | 42.50% | 42.65% | 7.35 pp | -41 | 23 | -1.78 |
| BTC Market Hours Daily | lstm | LSTM | 279 | 114 | 165 | 40.86% | 42.92% | 40.86% | 9.14 pp | -51 | 23 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Hourly | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Hourly | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 15 | -0.73 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 245 | 113 | 132 | 46.12% | 45.83% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 245 | 113 | 132 | 46.12% | 46.67% | 46.12% | 3.88 pp | -19 | 15 | -1.27 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 15 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 245 | 102 | 143 | 41.63% | 41.67% | 41.63% | 8.37 pp | -41 | 15 | -2.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 15 | -3.13 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 88 | 39 | 49 | 44.32% | 44.32% | 44.32% | 5.68 pp | -10 | 7 | -1.43 |
| Consolidated Market Hours Daily | rf | RandomForest | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 88 | 35 | 53 | 39.77% | 39.77% | 39.77% | 10.23 pp | -18 | 7 | -2.57 |
| Consolidated Market Hours Daily | lstm | LSTM | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 33 | 55 | 37.50% | 37.50% | 37.50% | 12.50 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours Daily | nn | NN | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 7 | -3.43 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
