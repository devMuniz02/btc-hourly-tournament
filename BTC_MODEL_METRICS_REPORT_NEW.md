# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T20:08:43.470967+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 199 | 139 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 235 | 175 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 19:00:00+00:00 | 314 | 163 | 151 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 19:00:00+00:00 | 314 | 163 | 151 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 137 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T14:00:00+00:00 | 138 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 6 | 0.50 |
| BTC Market Hours | nn | NN | 163 | 83 | 80 | 50.92% | 50.92% | 50.92% | 0.92 pp | 3 | 13 | 0.23 |
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| BTC Market Hours Daily | transformer | Transformer | 163 | 80 | 83 | 49.08% | 49.08% | 49.08% | 0.92 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| BTC Hourly | transformer | Transformer | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 6 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 163 | 78 | 85 | 47.85% | 47.85% | 47.85% | 2.15 pp | -7 | 14 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| BTC Market Hours | rf | RandomForest | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 13 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 14 | -1.07 |
| BTC Market Hours | transformer | Transformer | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 165 | 77 | 88 | 46.67% | 46.67% | 46.67% | 3.33 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 14 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 163 | 67 | 96 | 41.10% | 41.10% | 41.10% | 8.90 pp | -29 | 14 | -2.07 |
| BTC Market Hours | xgb | XGBoost | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 163 | 67 | 96 | 41.10% | 41.10% | 41.10% | 8.90 pp | -29 | 13 | -2.23 |
| BTC Daily | nn | NN | 165 | 73 | 92 | 44.24% | 44.24% | 44.24% | 5.76 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 163 | 63 | 100 | 38.65% | 38.65% | 38.65% | 11.35 pp | -37 | 14 | -2.64 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| BTC Daily | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 6 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 165 | 67 | 98 | 40.61% | 40.61% | 40.61% | 9.39 pp | -31 | 8 | -3.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 6 | -6.17 |
| BTC Daily | lstm | LSTM | 165 | 57 | 108 | 34.55% | 34.55% | 34.55% | 15.45 pp | -51 | 8 | -6.38 |
| BTC Hourly | lstm | LSTM | 139 | 50 | 89 | 35.97% | 35.97% | 35.97% | 14.03 pp | -39 | 6 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 139 | 71 | 68 | 51.08% | 51.08% | 51.08% | 1.08 pp | 3 | 6 | 0.50 |
| BTC Hourly | transformer | Transformer | 139 | 68 | 71 | 48.92% | 48.92% | 48.92% | 1.08 pp | -3 | 6 | -0.50 |
| BTC Hourly | nn | NN | 139 | 61 | 78 | 43.88% | 43.88% | 43.88% | 6.12 pp | -17 | 6 | -2.83 |
| BTC Hourly | rf | RandomForest | 139 | 59 | 80 | 42.45% | 42.45% | 42.45% | 7.55 pp | -21 | 6 | -3.50 |
| BTC Hourly | xgb | XGBoost | 139 | 51 | 88 | 36.69% | 36.69% | 36.69% | 13.31 pp | -37 | 6 | -6.17 |
| BTC Hourly | lstm | LSTM | 139 | 50 | 89 | 35.97% | 35.97% | 35.97% | 14.03 pp | -39 | 6 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 165 | 77 | 88 | 46.67% | 46.67% | 46.67% | 3.33 pp | -11 | 8 | -1.38 |
| BTC Daily | nn | NN | 165 | 73 | 92 | 44.24% | 44.24% | 44.24% | 5.76 pp | -19 | 8 | -2.38 |
| BTC Daily | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 8 | -3.12 |
| BTC Daily | rf | RandomForest | 165 | 67 | 98 | 40.61% | 40.61% | 40.61% | 9.39 pp | -31 | 8 | -3.88 |
| BTC Daily | xgb | XGBoost | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 9 | -5.44 |
| BTC Daily | lstm | LSTM | 165 | 57 | 108 | 34.55% | 34.55% | 34.55% | 15.45 pp | -51 | 8 | -6.38 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 163 | 83 | 80 | 50.92% | 50.92% | 50.92% | 0.92 pp | 3 | 13 | 0.23 |
| BTC Market Hours | rf | RandomForest | 163 | 75 | 88 | 46.01% | 46.01% | 46.01% | 3.99 pp | -13 | 13 | -1.00 |
| BTC Market Hours | transformer | Transformer | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 163 | 73 | 90 | 44.79% | 44.79% | 44.79% | 5.21 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 163 | 68 | 95 | 41.72% | 41.72% | 41.72% | 8.28 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 163 | 67 | 96 | 41.10% | 41.10% | 41.10% | 8.90 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 163 | 80 | 83 | 49.08% | 49.08% | 49.08% | 0.92 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 163 | 78 | 85 | 47.85% | 47.85% | 47.85% | 2.15 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 14 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 14 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 163 | 67 | 96 | 41.10% | 41.10% | 41.10% | 8.90 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 163 | 63 | 100 | 38.65% | 38.65% | 38.65% | 11.35 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 137 | 69 | 68 | 50.36% | 50.36% | 50.36% | 0.36 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 137 | 67 | 70 | 48.91% | 48.91% | 48.91% | 1.09 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 137 | 66 | 71 | 48.18% | 48.18% | 48.18% | 1.82 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 137 | 62 | 75 | 45.26% | 45.26% | 45.26% | 4.74 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 137 | 61 | 76 | 44.53% | 44.53% | 44.53% | 5.47 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 137 | 58 | 79 | 42.34% | 42.34% | 42.34% | 7.66 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 30 | 16 | 14 | 53.33% | 53.33% | 53.33% | 3.33 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 30 | 14 | 16 | 46.67% | 46.67% | 46.67% | 3.33 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 30 | 13 | 17 | 43.33% | 43.33% | 43.33% | 6.67 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 30 | 11 | 19 | 36.67% | 36.67% | 36.67% | 13.33 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 30 | 9 | 21 | 30.00% | 30.00% | 30.00% | 20.00 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 3 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 3 | -1.67 |
| Consolidated Market Hours Daily | nn | NN | 31 | 11 | 20 | 35.48% | 35.48% | 35.48% | 14.52 pp | -9 | 3 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 3 | -3.67 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
