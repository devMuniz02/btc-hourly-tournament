# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T13:58:03.129811+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 227 | 167 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 263 | 203 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 12:00:00+00:00 | 361 | 191 | 170 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 12:00:00+00:00 | 361 | 191 | 170 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 163 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 163 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 44 | 119 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 15:00:00+00:00 | 163 | 44 | 119 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 167 | 86 | 81 | 51.50% | 51.50% | 51.50% | 1.50 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 191 | 100 | 91 | 52.36% | 52.36% | 52.36% | 2.36 pp | 9 | 16 | 0.56 |
| BTC Market Hours | nn | NN | 191 | 98 | 93 | 51.31% | 51.31% | 51.31% | 1.31 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 191 | 92 | 99 | 48.17% | 48.17% | 48.17% | 1.83 pp | -7 | 16 | -0.44 |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 191 | 90 | 101 | 47.12% | 47.12% | 47.12% | 2.88 pp | -11 | 16 | -0.69 |
| BTC Hourly | transformer | Transformer | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 7 | -0.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 15 | -0.87 |
| BTC Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 191 | 85 | 106 | 44.50% | 44.50% | 44.50% | 5.50 pp | -21 | 16 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 191 | 83 | 108 | 43.46% | 43.46% | 43.46% | 6.54 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| BTC Market Hours | lstm | LSTM | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 191 | 79 | 112 | 41.36% | 41.36% | 41.36% | 8.64 pp | -33 | 16 | -2.06 |
| Consolidated Hourly | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 191 | 78 | 113 | 40.84% | 40.84% | 40.84% | 9.16 pp | -35 | 16 | -2.19 |
| BTC Daily | nn | NN | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 9 | -2.78 |
| BTC Hourly | nn | NN | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 193 | 76 | 117 | 39.38% | 39.38% | 39.38% | 10.62 pp | -41 | 9 | -4.56 |
| BTC Daily | xgb | XGBoost | 203 | 74 | 129 | 36.45% | 36.45% | 36.45% | 13.55 pp | -55 | 10 | -5.50 |
| BTC Hourly | lstm | LSTM | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 7 | -6.43 |
| BTC Daily | lstm | LSTM | 193 | 67 | 126 | 34.72% | 34.72% | 34.72% | 15.28 pp | -59 | 9 | -6.56 |
| BTC Hourly | xgb | XGBoost | 167 | 60 | 107 | 35.93% | 35.93% | 35.93% | 14.07 pp | -47 | 7 | -6.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 167 | 86 | 81 | 51.50% | 51.50% | 51.50% | 1.50 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 167 | 61 | 106 | 36.53% | 36.53% | 36.53% | 13.47 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 167 | 60 | 107 | 35.93% | 35.93% | 35.93% | 14.07 pp | -47 | 7 | -6.71 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 9 | -1.00 |
| BTC Daily | nn | NN | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 9 | -2.33 |
| BTC Daily | transformer | Transformer | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 9 | -2.78 |
| BTC Daily | rf | RandomForest | 193 | 76 | 117 | 39.38% | 39.38% | 39.38% | 10.62 pp | -41 | 9 | -4.56 |
| BTC Daily | xgb | XGBoost | 203 | 74 | 129 | 36.45% | 36.45% | 36.45% | 13.55 pp | -55 | 10 | -5.50 |
| BTC Daily | lstm | LSTM | 193 | 67 | 126 | 34.72% | 34.72% | 34.72% | 15.28 pp | -59 | 9 | -6.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 191 | 98 | 93 | 51.31% | 51.31% | 51.31% | 1.31 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 191 | 89 | 102 | 46.60% | 46.60% | 46.60% | 3.40 pp | -13 | 15 | -0.87 |
| BTC Market Hours | xgb | XGBoost | 191 | 83 | 108 | 43.46% | 43.46% | 43.46% | 6.54 pp | -25 | 15 | -1.67 |
| BTC Market Hours | lstm | LSTM | 191 | 82 | 109 | 42.93% | 42.93% | 42.93% | 7.07 pp | -27 | 15 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 191 | 100 | 91 | 52.36% | 52.36% | 52.36% | 2.36 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 191 | 92 | 99 | 48.17% | 48.17% | 48.17% | 1.83 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 191 | 90 | 101 | 47.12% | 47.12% | 47.12% | 2.88 pp | -11 | 16 | -0.69 |
| BTC Market Hours Daily | rf | RandomForest | 191 | 85 | 106 | 44.50% | 44.50% | 44.50% | 5.50 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 191 | 79 | 112 | 41.36% | 41.36% | 41.36% | 8.64 pp | -33 | 16 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 191 | 78 | 113 | 40.84% | 40.84% | 40.84% | 9.16 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 163 | 81 | 82 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 163 | 74 | 89 | 45.40% | 45.40% | 45.40% | 4.60 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 163 | 71 | 92 | 43.56% | 43.56% | 43.56% | 6.44 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 44 | 16 | 28 | 36.36% | 36.36% | 36.36% | 13.64 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
