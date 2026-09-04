# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T09:02:17.238822+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 208 | 148 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 244 | 184 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 328 | 172 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 327 | 171 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 145 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 18:00:00+00:00 | 145 | 34 | 111 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 7 | 0.86 |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 172 | 90 | 82 | 52.33% | 52.33% | 52.33% | 2.33 pp | 8 | 14 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 15 | -0.33 |
| BTC Hourly | transformer | Transformer | 148 | 72 | 76 | 48.65% | 48.65% | 48.65% | 1.35 pp | -4 | 7 | -0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | rf | RandomForest | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours Daily | nn | NN | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 15 | -0.87 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 14 | -1.71 |
| BTC Daily | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 15 | -1.93 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 14 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 171 | 67 | 104 | 39.18% | 39.18% | 39.18% | 10.82 pp | -37 | 15 | -2.47 |
| BTC Daily | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 8 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 148 | 64 | 84 | 43.24% | 43.24% | 43.24% | 6.76 pp | -20 | 7 | -2.86 |
| BTC Daily | transformer | Transformer | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 174 | 71 | 103 | 40.80% | 40.80% | 40.80% | 9.20 pp | -32 | 8 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 9 | -5.33 |
| BTC Hourly | lstm | LSTM | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 148 | 53 | 95 | 35.81% | 35.81% | 35.81% | 14.19 pp | -42 | 7 | -6.00 |
| BTC Daily | lstm | LSTM | 174 | 60 | 114 | 34.48% | 34.48% | 34.48% | 15.52 pp | -54 | 8 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 148 | 77 | 71 | 52.03% | 52.03% | 52.03% | 2.03 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 148 | 72 | 76 | 48.65% | 48.65% | 48.65% | 1.35 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 148 | 64 | 84 | 43.24% | 43.24% | 43.24% | 6.76 pp | -20 | 7 | -2.86 |
| BTC Hourly | rf | RandomForest | 148 | 61 | 87 | 41.22% | 41.22% | 41.22% | 8.78 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 148 | 54 | 94 | 36.49% | 36.49% | 36.49% | 13.51 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 148 | 53 | 95 | 35.81% | 35.81% | 35.81% | 14.19 pp | -42 | 7 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 174 | 80 | 94 | 45.98% | 45.98% | 45.98% | 4.02 pp | -14 | 8 | -1.75 |
| BTC Daily | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 8 | -2.50 |
| BTC Daily | transformer | Transformer | 174 | 75 | 99 | 43.10% | 43.10% | 43.10% | 6.90 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 174 | 71 | 103 | 40.80% | 40.80% | 40.80% | 9.20 pp | -32 | 8 | -4.00 |
| BTC Daily | xgb | XGBoost | 184 | 68 | 116 | 36.96% | 36.96% | 36.96% | 13.04 pp | -48 | 9 | -5.33 |
| BTC Daily | lstm | LSTM | 174 | 60 | 114 | 34.48% | 34.48% | 34.48% | 15.52 pp | -54 | 8 | -6.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 172 | 90 | 82 | 52.33% | 52.33% | 52.33% | 2.33 pp | 8 | 14 | 0.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | rf | RandomForest | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | transformer | Transformer | 172 | 80 | 92 | 46.51% | 46.51% | 46.51% | 3.49 pp | -12 | 14 | -0.86 |
| BTC Market Hours | xgb | XGBoost | 172 | 74 | 98 | 43.02% | 43.02% | 43.02% | 6.98 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 172 | 71 | 101 | 41.28% | 41.28% | 41.28% | 8.72 pp | -30 | 14 | -2.14 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 15 | -0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 171 | 79 | 92 | 46.20% | 46.20% | 46.20% | 3.80 pp | -13 | 15 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 171 | 71 | 100 | 41.52% | 41.52% | 41.52% | 8.48 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 171 | 67 | 104 | 39.18% | 39.18% | 39.18% | 10.82 pp | -37 | 15 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 145 | 76 | 69 | 52.41% | 52.41% | 52.41% | 2.41 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 145 | 69 | 76 | 47.59% | 47.59% | 47.59% | 2.41 pp | -7 | 11 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 145 | 68 | 77 | 46.90% | 46.90% | 46.90% | 3.10 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 145 | 67 | 78 | 46.21% | 46.21% | 46.21% | 3.79 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 145 | 62 | 83 | 42.76% | 42.76% | 42.76% | 7.24 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 145 | 61 | 84 | 42.07% | 42.07% | 42.07% | 7.93 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 34 | 16 | 18 | 47.06% | 47.06% | 47.06% | 2.94 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 34 | 11 | 23 | 32.35% | 32.35% | 32.35% | 17.65 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
