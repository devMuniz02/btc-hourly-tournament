# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T15:54:50.827354+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 294 | 234 | 60 | 0 |
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
| BTC Market Hours | nn | NN | 257 | 134 | 123 | 52.14% | 52.08% | 52.14% | 2.14 pp | 11 | 20 | 0.55 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 257 | 128 | 129 | 49.81% | 48.75% | 49.81% | 0.19 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | transformer | Transformer | 257 | 127 | 130 | 49.42% | 49.58% | 49.42% | 0.58 pp | -3 | 21 | -0.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 234 | 116 | 118 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | nn | NN | 257 | 125 | 132 | 48.64% | 48.33% | 48.64% | 1.36 pp | -7 | 21 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 257 | 125 | 132 | 48.64% | 48.75% | 48.64% | 1.36 pp | -7 | 20 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 225 | 110 | 115 | 48.89% | 48.89% | 48.89% | 1.11 pp | -5 | 14 | -0.36 |
| BTC Market Hours | rf | RandomForest | 257 | 121 | 136 | 47.08% | 45.83% | 47.08% | 2.92 pp | -15 | 20 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 259 | 123 | 136 | 47.49% | 47.50% | 47.49% | 2.51 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 225 | 104 | 121 | 46.22% | 46.22% | 46.22% | 3.78 pp | -17 | 14 | -1.21 |
| BTC Market Hours | transformer | Transformer | 257 | 116 | 141 | 45.14% | 45.00% | 45.14% | 4.86 pp | -25 | 20 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 257 | 114 | 143 | 44.36% | 44.17% | 44.36% | 5.64 pp | -29 | 21 | -1.38 |
| BTC Market Hours | xgb | XGBoost | 257 | 114 | 143 | 44.36% | 42.92% | 44.36% | 5.64 pp | -29 | 20 | -1.45 |
| BTC Market Hours Daily | rf | RandomForest | 257 | 113 | 144 | 43.97% | 42.92% | 43.97% | 6.03 pp | -31 | 21 | -1.48 |
| Consolidated Market Hours | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 77 | 33 | 44 | 42.86% | 42.86% | 42.86% | 7.14 pp | -11 | 6 | -1.83 |
| BTC Daily | nn | NN | 259 | 117 | 142 | 45.17% | 44.58% | 45.17% | 4.83 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 257 | 106 | 151 | 41.25% | 41.67% | 41.25% | 8.75 pp | -45 | 21 | -2.14 |
| Consolidated Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Hourly | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 225 | 97 | 128 | 43.11% | 43.11% | 43.11% | 6.89 pp | -31 | 14 | -2.21 |
| Consolidated Market Hours | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Hourly | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 225 | 91 | 134 | 40.44% | 40.44% | 40.44% | 9.56 pp | -43 | 14 | -3.07 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 77 | 29 | 48 | 37.66% | 37.66% | 37.66% | 12.34 pp | -19 | 6 | -3.17 |
| BTC Hourly | transformer | Transformer | 234 | 100 | 134 | 42.74% | 42.74% | 42.74% | 7.26 pp | -34 | 10 | -3.40 |
| BTC Market Hours | lstm | LSTM | 257 | 94 | 163 | 36.58% | 37.08% | 36.58% | 13.42 pp | -69 | 20 | -3.45 |
| Consolidated Market Hours | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 77 | 28 | 49 | 36.36% | 36.36% | 36.36% | 13.64 pp | -21 | 6 | -3.50 |
| BTC Hourly | nn | NN | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 10 | -3.60 |
| BTC Daily | transformer | Transformer | 259 | 104 | 155 | 40.15% | 39.17% | 40.15% | 9.85 pp | -51 | 12 | -4.25 |
| BTC Hourly | rf | RandomForest | 234 | 93 | 141 | 39.74% | 39.74% | 39.74% | 10.26 pp | -48 | 10 | -4.80 |
| BTC Hourly | lstm | LSTM | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 10 | -5.40 |
| BTC Daily | rf | RandomForest | 259 | 96 | 163 | 37.07% | 36.67% | 37.07% | 12.93 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 269 | 95 | 174 | 35.32% | 35.42% | 35.32% | 14.68 pp | -79 | 13 | -6.08 |
| BTC Daily | lstm | LSTM | 259 | 92 | 167 | 35.52% | 35.83% | 35.52% | 14.48 pp | -75 | 12 | -6.25 |
| BTC Hourly | xgb | XGBoost | 234 | 81 | 153 | 34.62% | 34.62% | 34.62% | 15.38 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 234 | 116 | 118 | 49.57% | 49.57% | 49.57% | 0.43 pp | -2 | 10 | -0.20 |
| BTC Hourly | transformer | Transformer | 234 | 100 | 134 | 42.74% | 42.74% | 42.74% | 7.26 pp | -34 | 10 | -3.40 |
| BTC Hourly | nn | NN | 234 | 99 | 135 | 42.31% | 42.31% | 42.31% | 7.69 pp | -36 | 10 | -3.60 |
| BTC Hourly | rf | RandomForest | 234 | 93 | 141 | 39.74% | 39.74% | 39.74% | 10.26 pp | -48 | 10 | -4.80 |
| BTC Hourly | lstm | LSTM | 234 | 90 | 144 | 38.46% | 38.46% | 38.46% | 11.54 pp | -54 | 10 | -5.40 |
| BTC Hourly | xgb | XGBoost | 234 | 81 | 153 | 34.62% | 34.62% | 34.62% | 15.38 pp | -72 | 10 | -7.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 259 | 123 | 136 | 47.49% | 47.50% | 47.49% | 2.51 pp | -13 | 12 | -1.08 |
| BTC Daily | nn | NN | 259 | 117 | 142 | 45.17% | 44.58% | 45.17% | 4.83 pp | -25 | 12 | -2.08 |
| BTC Daily | transformer | Transformer | 259 | 104 | 155 | 40.15% | 39.17% | 40.15% | 9.85 pp | -51 | 12 | -4.25 |
| BTC Daily | rf | RandomForest | 259 | 96 | 163 | 37.07% | 36.67% | 37.07% | 12.93 pp | -67 | 12 | -5.58 |
| BTC Daily | xgb | XGBoost | 269 | 95 | 174 | 35.32% | 35.42% | 35.32% | 14.68 pp | -79 | 13 | -6.08 |
| BTC Daily | lstm | LSTM | 259 | 92 | 167 | 35.52% | 35.83% | 35.52% | 14.48 pp | -75 | 12 | -6.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 257 | 134 | 123 | 52.14% | 52.08% | 52.14% | 2.14 pp | 11 | 20 | 0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 257 | 125 | 132 | 48.64% | 48.75% | 48.64% | 1.36 pp | -7 | 20 | -0.35 |
| BTC Market Hours | rf | RandomForest | 257 | 121 | 136 | 47.08% | 45.83% | 47.08% | 2.92 pp | -15 | 20 | -0.75 |
| BTC Market Hours | transformer | Transformer | 257 | 116 | 141 | 45.14% | 45.00% | 45.14% | 4.86 pp | -25 | 20 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 257 | 114 | 143 | 44.36% | 42.92% | 44.36% | 5.64 pp | -29 | 20 | -1.45 |
| BTC Market Hours | lstm | LSTM | 257 | 94 | 163 | 36.58% | 37.08% | 36.58% | 13.42 pp | -69 | 20 | -3.45 |

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
