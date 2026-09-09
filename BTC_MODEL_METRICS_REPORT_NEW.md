# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T15:47:01.431218+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 293 | 233 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 329 | 269 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 481 | 257 | 224 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 14:00:00+00:00 | 481 | 257 | 224 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 225 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 22:00:00+00:00 | 225 | 77 | 148 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 257 | 135 | 122 | 52.53% | 52.50% | 52.53% | 2.53 pp | 13 | 20 | 0.65 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 257 | 128 | 129 | 49.81% | 48.75% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 257 | 127 | 130 | 49.42% | 49.58% | 49.42% | 0.58 pp | -3 | 21 | -0.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| BTC Market Hours Daily | nn | NN | 257 | 125 | 132 | 48.64% | 48.33% | 48.64% | 1.36 pp | -7 | 21 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 257 | 122 | 135 | 47.47% | 47.50% | 47.47% | 2.53 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 20 | -0.95 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 257 | 116 | 141 | 45.14% | 43.75% | 45.14% | 4.86 pp | -25 | 20 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 257 | 114 | 143 | 44.36% | 44.17% | 44.36% | 5.64 pp | -29 | 21 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 257 | 113 | 144 | 43.97% | 42.92% | 43.97% | 6.03 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 259 | 120 | 139 | 46.33% | 45.83% | 46.33% | 3.67 pp | -19 | 12 | -1.58 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 259 | 117 | 142 | 45.17% | 44.17% | 45.17% | 4.83 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 257 | 106 | 151 | 41.25% | 41.67% | 41.25% | 8.75 pp | -45 | 21 | -2.14 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| BTC Market Hours | lstm | LSTM | 257 | 105 | 152 | 40.86% | 41.67% | 40.86% | 9.14 pp | -47 | 20 | -2.35 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| BTC Hourly | transformer | Transformer | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 10 | -2.70 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Hourly | nn | NN | 233 | 98 | 135 | 42.06% | 42.06% | 42.06% | 7.94 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 10 | -4.10 |
| BTC Daily | transformer | Transformer | 259 | 104 | 155 | 40.15% | 39.17% | 40.15% | 9.85 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 259 | 97 | 162 | 37.45% | 36.67% | 37.45% | 12.55 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 269 | 97 | 172 | 36.06% | 35.42% | 36.06% | 13.94 pp | -75 | 13 | -5.77 |
| BTC Hourly | lstm | LSTM | 233 | 87 | 146 | 37.34% | 37.34% | 37.34% | 12.66 pp | -59 | 10 | -5.90 |
| BTC Daily | lstm | LSTM | 259 | 91 | 168 | 35.14% | 35.83% | 35.14% | 14.86 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 233 | 80 | 153 | 34.33% | 34.33% | 34.33% | 15.67 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 233 | 115 | 118 | 49.36% | 49.36% | 49.36% | 0.64 pp | -3 | 10 | -0.30 |
| BTC Hourly | transformer | Transformer | 233 | 103 | 130 | 44.21% | 44.21% | 44.21% | 5.79 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 233 | 98 | 135 | 42.06% | 42.06% | 42.06% | 7.94 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 233 | 96 | 137 | 41.20% | 41.20% | 41.20% | 8.80 pp | -41 | 10 | -4.10 |
| BTC Hourly | lstm | LSTM | 233 | 87 | 146 | 37.34% | 37.34% | 37.34% | 12.66 pp | -59 | 10 | -5.90 |
| BTC Hourly | xgb | XGBoost | 233 | 80 | 153 | 34.33% | 34.33% | 34.33% | 15.67 pp | -73 | 10 | -7.30 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 259 | 120 | 139 | 46.33% | 45.83% | 46.33% | 3.67 pp | -19 | 12 | -1.58 |
| BTC Daily | nn | NN | 259 | 117 | 142 | 45.17% | 44.17% | 45.17% | 4.83 pp | -25 | 12 | -2.08 |
| BTC Daily | transformer | Transformer | 259 | 104 | 155 | 40.15% | 39.17% | 40.15% | 9.85 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 259 | 97 | 162 | 37.45% | 36.67% | 37.45% | 12.55 pp | -65 | 12 | -5.42 |
| BTC Daily | xgb | XGBoost | 269 | 97 | 172 | 36.06% | 35.42% | 36.06% | 13.94 pp | -75 | 13 | -5.77 |
| BTC Daily | lstm | LSTM | 259 | 91 | 168 | 35.14% | 35.83% | 35.14% | 14.86 pp | -77 | 12 | -6.42 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 257 | 135 | 122 | 52.53% | 52.50% | 52.53% | 2.53 pp | 13 | 20 | 0.65 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 257 | 122 | 135 | 47.47% | 47.50% | 47.47% | 2.53 pp | -13 | 20 | -0.65 |
| BTC Market Hours | transformer | Transformer | 257 | 121 | 136 | 47.08% | 47.08% | 47.08% | 2.92 pp | -15 | 20 | -0.75 |
| BTC Market Hours | xgb | XGBoost | 257 | 119 | 138 | 46.30% | 45.00% | 46.30% | 3.70 pp | -19 | 20 | -0.95 |
| BTC Market Hours | rf | RandomForest | 257 | 116 | 141 | 45.14% | 43.75% | 45.14% | 4.86 pp | -25 | 20 | -1.25 |
| BTC Market Hours | lstm | LSTM | 257 | 105 | 152 | 40.86% | 41.67% | 40.86% | 9.14 pp | -47 | 20 | -2.35 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 257 | 128 | 129 | 49.81% | 48.75% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 257 | 127 | 130 | 49.42% | 49.58% | 49.42% | 0.58 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | nn | NN | 257 | 125 | 132 | 48.64% | 48.33% | 48.64% | 1.36 pp | -7 | 21 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 257 | 114 | 143 | 44.36% | 44.17% | 44.36% | 5.64 pp | -29 | 21 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 257 | 113 | 144 | 43.97% | 42.92% | 43.97% | 6.03 pp | -31 | 21 | -1.48 |
| BTC Market Hours Daily | lstm | LSTM | 257 | 106 | 151 | 41.25% | 41.67% | 41.25% | 8.75 pp | -45 | 21 | -2.14 |

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
