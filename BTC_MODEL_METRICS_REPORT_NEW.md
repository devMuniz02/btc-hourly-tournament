# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T18:12:01.531671+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 247 | 187 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 283 | 223 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 399 | 211 | 188 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 17:00:00+00:00 | 398 | 210 | 188 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 181 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 12:00:00+00:00 | 181 | 54 | 127 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Market Hours | nn | NN | 211 | 109 | 102 | 51.66% | 51.66% | 51.66% | 1.66 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| BTC Market Hours | transformer | Transformer | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 17 | -0.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 211 | 99 | 112 | 46.92% | 46.92% | 46.92% | 3.08 pp | -13 | 17 | -0.76 |
| BTC Daily | mlp_sklearn | MLPClassifier | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| BTC Market Hours | rf | RandomForest | 211 | 97 | 114 | 45.97% | 45.97% | 45.97% | 4.03 pp | -17 | 17 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 17 | -1.59 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| BTC Hourly | transformer | Transformer | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |
| BTC Daily | nn | NN | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 10 | -2.10 |
| BTC Market Hours | lstm | LSTM | 211 | 87 | 124 | 41.23% | 41.23% | 41.23% | 8.77 pp | -37 | 17 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Hourly | nn | NN | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 8 | -3.88 |
| BTC Daily | rf | RandomForest | 213 | 81 | 132 | 38.03% | 38.03% | 38.03% | 11.97 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 223 | 80 | 143 | 35.87% | 35.87% | 35.87% | 14.13 pp | -63 | 11 | -5.73 |
| BTC Hourly | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 8 | -6.38 |
| BTC Daily | lstm | LSTM | 213 | 71 | 142 | 33.33% | 33.33% | 33.33% | 16.67 pp | -71 | 10 | -7.10 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 187 | 94 | 93 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 187 | 86 | 101 | 45.99% | 45.99% | 45.99% | 4.01 pp | -15 | 8 | -1.88 |
| BTC Hourly | nn | NN | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 8 | -3.88 |
| BTC Hourly | xgb | XGBoost | 187 | 69 | 118 | 36.90% | 36.90% | 36.90% | 13.10 pp | -49 | 8 | -6.12 |
| BTC Hourly | lstm | LSTM | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 8 | -6.38 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 213 | 102 | 111 | 47.89% | 47.89% | 47.89% | 2.11 pp | -9 | 10 | -0.90 |
| BTC Daily | nn | NN | 213 | 96 | 117 | 45.07% | 45.07% | 45.07% | 4.93 pp | -21 | 10 | -2.10 |
| BTC Daily | transformer | Transformer | 213 | 89 | 124 | 41.78% | 41.78% | 41.78% | 8.22 pp | -35 | 10 | -3.50 |
| BTC Daily | rf | RandomForest | 213 | 81 | 132 | 38.03% | 38.03% | 38.03% | 11.97 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 223 | 80 | 143 | 35.87% | 35.87% | 35.87% | 14.13 pp | -63 | 11 | -5.73 |
| BTC Daily | lstm | LSTM | 213 | 71 | 142 | 33.33% | 33.33% | 33.33% | 16.67 pp | -71 | 10 | -7.10 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 211 | 109 | 102 | 51.66% | 51.66% | 51.66% | 1.66 pp | 7 | 17 | 0.41 |
| BTC Market Hours | transformer | Transformer | 211 | 103 | 108 | 48.82% | 48.82% | 48.82% | 1.18 pp | -5 | 17 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 211 | 99 | 112 | 46.92% | 46.92% | 46.92% | 3.08 pp | -13 | 17 | -0.76 |
| BTC Market Hours | rf | RandomForest | 211 | 97 | 114 | 45.97% | 45.97% | 45.97% | 4.03 pp | -17 | 17 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 211 | 92 | 119 | 43.60% | 43.60% | 43.60% | 6.40 pp | -27 | 17 | -1.59 |
| BTC Market Hours | lstm | LSTM | 211 | 87 | 124 | 41.23% | 41.23% | 41.23% | 8.77 pp | -37 | 17 | -2.18 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 210 | 109 | 101 | 51.90% | 51.90% | 51.90% | 1.90 pp | 8 | 18 | 0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 210 | 104 | 106 | 49.52% | 49.52% | 49.52% | 0.48 pp | -2 | 18 | -0.11 |
| BTC Market Hours Daily | nn | NN | 210 | 101 | 109 | 48.10% | 48.10% | 48.10% | 1.90 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | rf | RandomForest | 210 | 93 | 117 | 44.29% | 44.29% | 44.29% | 5.71 pp | -24 | 18 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 210 | 87 | 123 | 41.43% | 41.43% | 41.43% | 8.57 pp | -36 | 18 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 210 | 84 | 126 | 40.00% | 40.00% | 40.00% | 10.00 pp | -42 | 18 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 181 | 89 | 92 | 49.17% | 49.17% | 49.17% | 0.83 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 181 | 88 | 93 | 48.62% | 48.62% | 48.62% | 1.38 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 181 | 84 | 97 | 46.41% | 46.41% | 46.41% | 3.59 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 181 | 79 | 102 | 43.65% | 43.65% | 43.65% | 6.35 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 181 | 77 | 104 | 42.54% | 42.54% | 42.54% | 7.46 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
