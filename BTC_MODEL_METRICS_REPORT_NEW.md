# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T07:29:57.356669+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 223 | 163 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 259 | 199 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 356 | 187 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 356 | 187 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 159 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 159 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 159 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T13:00:00+00:00 | 160 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 163 | 84 | 79 | 51.53% | 51.53% | 51.53% | 1.53 pp | 5 | 7 | 0.71 |
| BTC Market Hours Daily | transformer | Transformer | 187 | 98 | 89 | 52.41% | 52.41% | 52.41% | 2.41 pp | 9 | 16 | 0.56 |
| BTC Market Hours | nn | NN | 187 | 96 | 91 | 51.34% | 51.34% | 51.34% | 1.34 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 187 | 93 | 94 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 187 | 90 | 97 | 48.13% | 48.13% | 48.13% | 1.87 pp | -7 | 16 | -0.44 |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 7 | -0.71 |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 16 | -0.81 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 15 | -0.87 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| BTC Daily | mlp_sklearn | MLPClassifier | 189 | 89 | 100 | 47.09% | 47.09% | 47.09% | 2.91 pp | -11 | 9 | -1.22 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | lstm | LSTM | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 159 | 70 | 89 | 44.03% | 44.03% | 44.03% | 5.97 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 70 | 89 | 44.03% | 44.03% | 44.03% | 5.97 pp | -19 | 12 | -1.58 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 15 | -1.80 |
| BTC Market Hours | lstm | LSTM | 187 | 79 | 108 | 42.25% | 42.25% | 42.25% | 7.75 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | xgb | XGBoost | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 16 | -1.94 |
| Consolidated Hourly | transformer | Transformer | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 187 | 75 | 112 | 40.11% | 40.11% | 40.11% | 9.89 pp | -37 | 16 | -2.31 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| BTC Daily | nn | NN | 189 | 83 | 106 | 43.92% | 43.92% | 43.92% | 6.08 pp | -23 | 9 | -2.56 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 9 | -2.78 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 189 | 74 | 115 | 39.15% | 39.15% | 39.15% | 10.85 pp | -41 | 9 | -4.56 |
| BTC Daily | xgb | XGBoost | 199 | 73 | 126 | 36.68% | 36.68% | 36.68% | 13.32 pp | -53 | 10 | -5.30 |
| BTC Hourly | lstm | LSTM | 163 | 60 | 103 | 36.81% | 36.81% | 36.81% | 13.19 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 189 | 66 | 123 | 34.92% | 34.92% | 34.92% | 15.08 pp | -57 | 9 | -6.33 |
| BTC Hourly | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 7 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 163 | 84 | 79 | 51.53% | 51.53% | 51.53% | 1.53 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 163 | 79 | 84 | 48.47% | 48.47% | 48.47% | 1.53 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 163 | 70 | 93 | 42.94% | 42.94% | 42.94% | 7.06 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 163 | 69 | 94 | 42.33% | 42.33% | 42.33% | 7.67 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 163 | 60 | 103 | 36.81% | 36.81% | 36.81% | 13.19 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 163 | 57 | 106 | 34.97% | 34.97% | 34.97% | 15.03 pp | -49 | 7 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 189 | 89 | 100 | 47.09% | 47.09% | 47.09% | 2.91 pp | -11 | 9 | -1.22 |
| BTC Daily | nn | NN | 189 | 83 | 106 | 43.92% | 43.92% | 43.92% | 6.08 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 189 | 82 | 107 | 43.39% | 43.39% | 43.39% | 6.61 pp | -25 | 9 | -2.78 |
| BTC Daily | rf | RandomForest | 189 | 74 | 115 | 39.15% | 39.15% | 39.15% | 10.85 pp | -41 | 9 | -4.56 |
| BTC Daily | xgb | XGBoost | 199 | 73 | 126 | 36.68% | 36.68% | 36.68% | 13.32 pp | -53 | 10 | -5.30 |
| BTC Daily | lstm | LSTM | 189 | 66 | 123 | 34.92% | 34.92% | 34.92% | 15.08 pp | -57 | 9 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 187 | 96 | 91 | 51.34% | 51.34% | 51.34% | 1.34 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 187 | 93 | 94 | 49.73% | 49.73% | 49.73% | 0.27 pp | -1 | 15 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 15 | -0.87 |
| BTC Market Hours | xgb | XGBoost | 187 | 80 | 107 | 42.78% | 42.78% | 42.78% | 7.22 pp | -27 | 15 | -1.80 |
| BTC Market Hours | lstm | LSTM | 187 | 79 | 108 | 42.25% | 42.25% | 42.25% | 7.75 pp | -29 | 15 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 187 | 98 | 89 | 52.41% | 52.41% | 52.41% | 2.41 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 187 | 90 | 97 | 48.13% | 48.13% | 48.13% | 1.87 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 187 | 87 | 100 | 46.52% | 46.52% | 46.52% | 3.48 pp | -13 | 16 | -0.81 |
| BTC Market Hours Daily | rf | RandomForest | 187 | 83 | 104 | 44.39% | 44.39% | 44.39% | 5.61 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | xgb | XGBoost | 187 | 78 | 109 | 41.71% | 41.71% | 41.71% | 8.29 pp | -31 | 16 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 187 | 75 | 112 | 40.11% | 40.11% | 40.11% | 9.89 pp | -37 | 16 | -2.31 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 159 | 70 | 89 | 44.03% | 44.03% | 44.03% | 5.97 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 159 | 78 | 81 | 49.06% | 49.06% | 49.06% | 0.94 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 159 | 77 | 82 | 48.43% | 48.43% | 48.43% | 1.57 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 159 | 73 | 86 | 45.91% | 45.91% | 45.91% | 4.09 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 159 | 71 | 88 | 44.65% | 44.65% | 44.65% | 5.35 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 159 | 70 | 89 | 44.03% | 44.03% | 44.03% | 5.97 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 159 | 67 | 92 | 42.14% | 42.14% | 42.14% | 7.86 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | transformer | Transformer | 42 | 16 | 26 | 38.10% | 38.10% | 38.10% | 11.90 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 42 | 15 | 27 | 35.71% | 35.71% | 35.71% | 14.29 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
