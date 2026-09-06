# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T11:34:17.057127+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 242 | 182 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 278 | 218 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 388 | 206 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 388 | 206 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T23:00:00+00:00 | 178 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T23:00:00+00:00 | 178 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T23:00:00+00:00 | 178 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T23:00:00+00:00 | 179 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 206 | 107 | 99 | 51.94% | 51.94% | 51.94% | 1.94 pp | 8 | 16 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 206 | 106 | 100 | 51.46% | 51.46% | 51.46% | 1.46 pp | 6 | 17 | 0.35 |
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 4 | 0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 182 | 92 | 90 | 50.55% | 50.55% | 50.55% | 0.55 pp | 2 | 8 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 17 | -0.24 |
| BTC Market Hours | transformer | Transformer | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 16 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | nn | NN | 206 | 98 | 108 | 47.57% | 47.57% | 47.57% | 2.43 pp | -10 | 17 | -0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 206 | 97 | 109 | 47.09% | 47.09% | 47.09% | 2.91 pp | -12 | 16 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 9 | -0.89 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 206 | 94 | 112 | 45.63% | 45.63% | 45.63% | 4.37 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 12 | -1.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 12 | -1.33 |
| BTC Hourly | transformer | Transformer | 182 | 85 | 97 | 46.70% | 46.70% | 46.70% | 3.30 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 17 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 16 | -1.62 |
| Consolidated Hourly | nn | NN | 178 | 79 | 99 | 44.38% | 44.38% | 44.38% | 5.62 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 79 | 99 | 44.38% | 44.38% | 44.38% | 5.62 pp | -20 | 12 | -1.67 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 206 | 86 | 120 | 41.75% | 41.75% | 41.75% | 8.25 pp | -34 | 16 | -2.12 |
| Consolidated Hourly | transformer | Transformer | 178 | 76 | 102 | 42.70% | 42.70% | 42.70% | 7.30 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 76 | 102 | 42.70% | 42.70% | 42.70% | 7.30 pp | -26 | 12 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 206 | 84 | 122 | 40.78% | 40.78% | 40.78% | 9.22 pp | -38 | 17 | -2.24 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 206 | 83 | 123 | 40.29% | 40.29% | 40.29% | 9.71 pp | -40 | 17 | -2.35 |
| BTC Daily | nn | NN | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 4 | -2.75 |
| BTC Hourly | nn | NN | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |
| BTC Daily | transformer | Transformer | 208 | 87 | 121 | 41.83% | 41.83% | 41.83% | 8.17 pp | -34 | 9 | -3.78 |
| BTC Hourly | xgb | XGBoost | 182 | 68 | 114 | 37.36% | 37.36% | 37.36% | 12.64 pp | -46 | 8 | -5.75 |
| BTC Daily | rf | RandomForest | 208 | 78 | 130 | 37.50% | 37.50% | 37.50% | 12.50 pp | -52 | 9 | -5.78 |
| BTC Hourly | lstm | LSTM | 182 | 67 | 115 | 36.81% | 36.81% | 36.81% | 13.19 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 218 | 79 | 139 | 36.24% | 36.24% | 36.24% | 13.76 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 208 | 70 | 138 | 33.65% | 33.65% | 33.65% | 16.35 pp | -68 | 9 | -7.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 182 | 92 | 90 | 50.55% | 50.55% | 50.55% | 0.55 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 182 | 85 | 97 | 46.70% | 46.70% | 46.70% | 3.30 pp | -12 | 8 | -1.50 |
| BTC Hourly | nn | NN | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 8 | -3.50 |
| BTC Hourly | xgb | XGBoost | 182 | 68 | 114 | 37.36% | 37.36% | 37.36% | 12.64 pp | -46 | 8 | -5.75 |
| BTC Hourly | lstm | LSTM | 182 | 67 | 115 | 36.81% | 36.81% | 36.81% | 13.19 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 208 | 100 | 108 | 48.08% | 48.08% | 48.08% | 1.92 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 208 | 93 | 115 | 44.71% | 44.71% | 44.71% | 5.29 pp | -22 | 9 | -2.44 |
| BTC Daily | transformer | Transformer | 208 | 87 | 121 | 41.83% | 41.83% | 41.83% | 8.17 pp | -34 | 9 | -3.78 |
| BTC Daily | rf | RandomForest | 208 | 78 | 130 | 37.50% | 37.50% | 37.50% | 12.50 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 218 | 79 | 139 | 36.24% | 36.24% | 36.24% | 13.76 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 208 | 70 | 138 | 33.65% | 33.65% | 33.65% | 16.35 pp | -68 | 9 | -7.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 206 | 107 | 99 | 51.94% | 51.94% | 51.94% | 1.94 pp | 8 | 16 | 0.50 |
| BTC Market Hours | transformer | Transformer | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 206 | 97 | 109 | 47.09% | 47.09% | 47.09% | 2.91 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 206 | 94 | 112 | 45.63% | 45.63% | 45.63% | 4.37 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 16 | -1.62 |
| BTC Market Hours | lstm | LSTM | 206 | 86 | 120 | 41.75% | 41.75% | 41.75% | 8.25 pp | -34 | 16 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 206 | 106 | 100 | 51.46% | 51.46% | 51.46% | 1.46 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 206 | 101 | 105 | 49.03% | 49.03% | 49.03% | 0.97 pp | -4 | 17 | -0.24 |
| BTC Market Hours Daily | nn | NN | 206 | 98 | 108 | 47.57% | 47.57% | 47.57% | 2.43 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 206 | 90 | 116 | 43.69% | 43.69% | 43.69% | 6.31 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 206 | 84 | 122 | 40.78% | 40.78% | 40.78% | 9.22 pp | -38 | 17 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 206 | 83 | 123 | 40.29% | 40.29% | 40.29% | 9.71 pp | -40 | 17 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 12 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | lstm | LSTM | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | nn | NN | 178 | 79 | 99 | 44.38% | 44.38% | 44.38% | 5.62 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 178 | 76 | 102 | 42.70% | 42.70% | 42.70% | 7.30 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 178 | 79 | 99 | 44.38% | 44.38% | 44.38% | 5.62 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 178 | 76 | 102 | 42.70% | 42.70% | 42.70% | 7.30 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 52 | 24 | 28 | 46.15% | 46.15% | 46.15% | 3.85 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 52 | 21 | 31 | 40.38% | 40.38% | 40.38% | 9.62 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 4 | 0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
