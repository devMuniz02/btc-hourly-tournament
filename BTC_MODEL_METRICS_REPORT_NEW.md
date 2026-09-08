# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T21:49:19.425704+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 281 | 221 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 317 | 257 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 20:00:00+00:00 | 462 | 245 | 217 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 20:00:00+00:00 | 462 | 245 | 217 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 213 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-31T16:00:00+00:00 | 214 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 245 | 125 | 120 | 51.02% | 51.67% | 51.02% | 1.02 pp | 5 | 19 | 0.26 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 245 | 120 | 125 | 48.98% | 48.75% | 48.98% | 1.02 pp | -5 | 20 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 221 | 109 | 112 | 49.32% | 49.32% | 49.32% | 0.68 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | transformer | Transformer | 245 | 119 | 126 | 48.57% | 48.33% | 48.57% | 1.43 pp | -7 | 20 | -0.35 |
| BTC Market Hours Daily | nn | NN | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 20 | -0.55 |
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 245 | 116 | 129 | 47.35% | 47.50% | 47.35% | 2.65 pp | -13 | 19 | -0.68 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 245 | 114 | 131 | 46.53% | 47.08% | 46.53% | 3.47 pp | -17 | 19 | -0.89 |
| BTC Market Hours | xgb | XGBoost | 245 | 114 | 131 | 46.53% | 45.83% | 46.53% | 3.47 pp | -17 | 19 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 245 | 110 | 135 | 44.90% | 44.58% | 44.90% | 5.10 pp | -25 | 19 | -1.32 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 20 | -1.35 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 247 | 115 | 132 | 46.56% | 46.67% | 46.56% | 3.44 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 20 | -1.55 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 247 | 113 | 134 | 45.75% | 45.42% | 45.75% | 4.25 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 245 | 104 | 141 | 42.45% | 42.92% | 42.45% | 7.55 pp | -37 | 19 | -1.95 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 20 | -2.35 |
| BTC Hourly | transformer | Transformer | 221 | 98 | 123 | 44.34% | 44.34% | 44.34% | 5.66 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 221 | 92 | 129 | 41.63% | 41.63% | 41.63% | 8.37 pp | -37 | 10 | -3.70 |
| BTC Daily | transformer | Transformer | 247 | 99 | 148 | 40.08% | 39.17% | 40.08% | 9.92 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 247 | 95 | 152 | 38.46% | 37.50% | 38.46% | 11.54 pp | -57 | 11 | -5.18 |
| BTC Hourly | lstm | LSTM | 221 | 82 | 139 | 37.10% | 37.10% | 37.10% | 12.90 pp | -57 | 10 | -5.70 |
| BTC Daily | xgb | XGBoost | 257 | 91 | 166 | 35.41% | 35.00% | 35.41% | 14.59 pp | -75 | 12 | -6.25 |
| BTC Hourly | xgb | XGBoost | 221 | 77 | 144 | 34.84% | 34.84% | 34.84% | 15.16 pp | -67 | 10 | -6.70 |
| BTC Daily | lstm | LSTM | 247 | 84 | 163 | 34.01% | 34.17% | 34.01% | 15.99 pp | -79 | 11 | -7.18 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 221 | 109 | 112 | 49.32% | 49.32% | 49.32% | 0.68 pp | -3 | 10 | -0.30 |
| BTC Hourly | transformer | Transformer | 221 | 98 | 123 | 44.34% | 44.34% | 44.34% | 5.66 pp | -25 | 10 | -2.50 |
| BTC Hourly | nn | NN | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 221 | 92 | 129 | 41.63% | 41.63% | 41.63% | 8.37 pp | -37 | 10 | -3.70 |
| BTC Hourly | lstm | LSTM | 221 | 82 | 139 | 37.10% | 37.10% | 37.10% | 12.90 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 221 | 77 | 144 | 34.84% | 34.84% | 34.84% | 15.16 pp | -67 | 10 | -6.70 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 247 | 115 | 132 | 46.56% | 46.67% | 46.56% | 3.44 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 247 | 113 | 134 | 45.75% | 45.42% | 45.75% | 4.25 pp | -21 | 11 | -1.91 |
| BTC Daily | transformer | Transformer | 247 | 99 | 148 | 40.08% | 39.17% | 40.08% | 9.92 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 247 | 95 | 152 | 38.46% | 37.50% | 38.46% | 11.54 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 257 | 91 | 166 | 35.41% | 35.00% | 35.41% | 14.59 pp | -75 | 12 | -6.25 |
| BTC Daily | lstm | LSTM | 247 | 84 | 163 | 34.01% | 34.17% | 34.01% | 15.99 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 245 | 125 | 120 | 51.02% | 51.67% | 51.02% | 1.02 pp | 5 | 19 | 0.26 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 245 | 116 | 129 | 47.35% | 47.50% | 47.35% | 2.65 pp | -13 | 19 | -0.68 |
| BTC Market Hours | transformer | Transformer | 245 | 114 | 131 | 46.53% | 47.08% | 46.53% | 3.47 pp | -17 | 19 | -0.89 |
| BTC Market Hours | xgb | XGBoost | 245 | 114 | 131 | 46.53% | 45.83% | 46.53% | 3.47 pp | -17 | 19 | -0.89 |
| BTC Market Hours | rf | RandomForest | 245 | 110 | 135 | 44.90% | 44.58% | 44.90% | 5.10 pp | -25 | 19 | -1.32 |
| BTC Market Hours | lstm | LSTM | 245 | 104 | 141 | 42.45% | 42.92% | 42.45% | 7.55 pp | -37 | 19 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 245 | 120 | 125 | 48.98% | 48.75% | 48.98% | 1.02 pp | -5 | 20 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 245 | 119 | 126 | 48.57% | 48.33% | 48.57% | 1.43 pp | -7 | 20 | -0.35 |
| BTC Market Hours Daily | nn | NN | 245 | 117 | 128 | 47.76% | 47.92% | 47.76% | 2.24 pp | -11 | 20 | -0.55 |
| BTC Market Hours Daily | xgb | XGBoost | 245 | 109 | 136 | 44.49% | 43.75% | 44.49% | 5.51 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | rf | RandomForest | 245 | 107 | 138 | 43.67% | 42.92% | 43.67% | 6.33 pp | -31 | 20 | -1.55 |
| BTC Market Hours Daily | lstm | LSTM | 245 | 99 | 146 | 40.41% | 40.83% | 40.41% | 9.59 pp | -47 | 20 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Hourly | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 213 | 101 | 112 | 47.42% | 47.42% | 47.42% | 2.58 pp | -11 | 14 | -0.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 213 | 98 | 115 | 46.01% | 46.01% | 46.01% | 3.99 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 213 | 97 | 116 | 45.54% | 45.54% | 45.54% | 4.46 pp | -19 | 14 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 213 | 93 | 120 | 43.66% | 43.66% | 43.66% | 6.34 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 213 | 91 | 122 | 42.72% | 42.72% | 42.72% | 7.28 pp | -31 | 14 | -2.21 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 71 | 33 | 38 | 46.48% | 46.48% | 46.48% | 3.52 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 72 | 34 | 38 | 47.22% | 47.22% | 47.22% | 2.78 pp | -4 | 6 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 6 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 6 | -1.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 72 | 29 | 43 | 40.28% | 40.28% | 40.28% | 9.72 pp | -14 | 6 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 6 | -2.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
