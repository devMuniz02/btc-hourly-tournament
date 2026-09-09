# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T14:20:15.213311+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 292 | 232 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 328 | 268 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 13:00:00+00:00 | 479 | 256 | 223 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 13:00:00+00:00 | 479 | 256 | 223 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 120 | 138 | 46.51% | 46.25% | 46.51% | 3.49 pp | -18 | 12 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 258 | 117 | 141 | 45.35% | 44.17% | 45.35% | 4.65 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Hourly | transformer | Transformer | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 10 | -2.60 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Hourly | nn | NN | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 10 | -4.00 |
| BTC Daily | transformer | Transformer | 258 | 104 | 154 | 40.31% | 39.58% | 40.31% | 9.69 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 258 | 97 | 161 | 37.60% | 36.67% | 37.60% | 12.40 pp | -64 | 12 | -5.33 |
| BTC Hourly | lstm | LSTM | 232 | 87 | 145 | 37.50% | 37.50% | 37.50% | 12.50 pp | -58 | 10 | -5.80 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.00% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 258 | 90 | 168 | 34.88% | 35.42% | 34.88% | 15.12 pp | -78 | 12 | -6.50 |
| BTC Hourly | xgb | XGBoost | 232 | 80 | 152 | 34.48% | 34.48% | 34.48% | 15.52 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 232 | 115 | 117 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Hourly | transformer | Transformer | 232 | 103 | 129 | 44.40% | 44.40% | 44.40% | 5.60 pp | -26 | 10 | -2.60 |
| BTC Hourly | nn | NN | 232 | 98 | 134 | 42.24% | 42.24% | 42.24% | 7.76 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 232 | 96 | 136 | 41.38% | 41.38% | 41.38% | 8.62 pp | -40 | 10 | -4.00 |
| BTC Hourly | lstm | LSTM | 232 | 87 | 145 | 37.50% | 37.50% | 37.50% | 12.50 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 232 | 80 | 152 | 34.48% | 34.48% | 34.48% | 15.52 pp | -72 | 10 | -7.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 258 | 120 | 138 | 46.51% | 46.25% | 46.51% | 3.49 pp | -18 | 12 | -1.50 |
| BTC Daily | nn | NN | 258 | 117 | 141 | 45.35% | 44.17% | 45.35% | 4.65 pp | -24 | 12 | -2.00 |
| BTC Daily | transformer | Transformer | 258 | 104 | 154 | 40.31% | 39.58% | 40.31% | 9.69 pp | -50 | 12 | -4.17 |
| BTC Daily | rf | RandomForest | 258 | 97 | 161 | 37.60% | 36.67% | 37.60% | 12.40 pp | -64 | 12 | -5.33 |
| BTC Daily | xgb | XGBoost | 268 | 96 | 172 | 35.82% | 35.00% | 35.82% | 14.18 pp | -76 | 13 | -5.85 |
| BTC Daily | lstm | LSTM | 258 | 90 | 168 | 34.88% | 35.42% | 34.88% | 15.12 pp | -78 | 12 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 256 | 135 | 121 | 52.73% | 52.92% | 52.73% | 2.73 pp | 14 | 20 | 0.70 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 256 | 121 | 135 | 47.27% | 47.08% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 256 | 121 | 135 | 47.27% | 47.50% | 47.27% | 2.73 pp | -14 | 20 | -0.70 |
| BTC Market Hours | xgb | XGBoost | 256 | 119 | 137 | 46.48% | 45.42% | 46.48% | 3.52 pp | -18 | 20 | -0.90 |
| BTC Market Hours | rf | RandomForest | 256 | 116 | 140 | 45.31% | 44.17% | 45.31% | 4.69 pp | -24 | 20 | -1.20 |
| BTC Market Hours | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.67% | 41.02% | 8.98 pp | -46 | 20 | -2.30 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 256 | 127 | 129 | 49.61% | 48.75% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | transformer | Transformer | 256 | 127 | 129 | 49.61% | 50.00% | 49.61% | 0.39 pp | -2 | 21 | -0.10 |
| BTC Market Hours Daily | nn | NN | 256 | 125 | 131 | 48.83% | 48.75% | 48.83% | 1.17 pp | -6 | 21 | -0.29 |
| BTC Market Hours Daily | xgb | XGBoost | 256 | 114 | 142 | 44.53% | 44.58% | 44.53% | 5.47 pp | -28 | 21 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 256 | 112 | 144 | 43.75% | 42.92% | 43.75% | 6.25 pp | -32 | 21 | -1.52 |
| BTC Market Hours Daily | lstm | LSTM | 256 | 105 | 151 | 41.02% | 41.25% | 41.02% | 8.98 pp | -46 | 21 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
