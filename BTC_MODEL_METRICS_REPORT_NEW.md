# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T14:53:10.931061+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 212 | 152 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 247 | 187 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 333 | 175 | 158 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 13:00:00+00:00 | 333 | 175 | 158 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 149 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 149 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 36 | 113 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 20:00:00+00:00 | 149 | 36 | 113 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 7 | 0.86 |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 175 | 92 | 83 | 52.57% | 52.57% | 52.57% | 2.57 pp | 9 | 14 | 0.64 |
| Consolidated Hourly | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| BTC Market Hours Daily | transformer | Transformer | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 15 | 0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 15 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| BTC Hourly | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| BTC Market Hours | transformer | Transformer | 175 | 83 | 92 | 47.43% | 47.43% | 47.43% | 2.57 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | nn | NN | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 14 | -0.93 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | rf | RandomForest | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 15 | -1.27 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 175 | 75 | 100 | 42.86% | 42.86% | 42.86% | 7.14 pp | -25 | 14 | -1.79 |
| BTC Daily | mlp_sklearn | MLPClassifier | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 15 | -1.93 |
| BTC Market Hours | lstm | LSTM | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 14 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 175 | 70 | 105 | 40.00% | 40.00% | 40.00% | 10.00 pp | -35 | 15 | -2.33 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |
| BTC Daily | nn | NN | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 8 | -2.88 |
| BTC Daily | transformer | Transformer | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 8 | -3.12 |
| BTC Hourly | nn | NN | 152 | 65 | 87 | 42.76% | 42.76% | 42.76% | 7.24 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 152 | 62 | 90 | 40.79% | 40.79% | 40.79% | 9.21 pp | -28 | 7 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 177 | 71 | 106 | 40.11% | 40.11% | 40.11% | 9.89 pp | -35 | 8 | -4.38 |
| BTC Daily | xgb | XGBoost | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 9 | -5.67 |
| BTC Hourly | lstm | LSTM | 152 | 55 | 97 | 36.18% | 36.18% | 36.18% | 13.82 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 7 | -6.29 |
| BTC Daily | lstm | LSTM | 177 | 62 | 115 | 35.03% | 35.03% | 35.03% | 14.97 pp | -53 | 8 | -6.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 152 | 79 | 73 | 51.97% | 51.97% | 51.97% | 1.97 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 152 | 74 | 78 | 48.68% | 48.68% | 48.68% | 1.32 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 152 | 65 | 87 | 42.76% | 42.76% | 42.76% | 7.24 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 152 | 62 | 90 | 40.79% | 40.79% | 40.79% | 9.21 pp | -28 | 7 | -4.00 |
| BTC Hourly | lstm | LSTM | 152 | 55 | 97 | 36.18% | 36.18% | 36.18% | 13.82 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 152 | 54 | 98 | 35.53% | 35.53% | 35.53% | 14.47 pp | -44 | 7 | -6.29 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 177 | 81 | 96 | 45.76% | 45.76% | 45.76% | 4.24 pp | -15 | 8 | -1.88 |
| BTC Daily | nn | NN | 177 | 77 | 100 | 43.50% | 43.50% | 43.50% | 6.50 pp | -23 | 8 | -2.88 |
| BTC Daily | transformer | Transformer | 177 | 76 | 101 | 42.94% | 42.94% | 42.94% | 7.06 pp | -25 | 8 | -3.12 |
| BTC Daily | rf | RandomForest | 177 | 71 | 106 | 40.11% | 40.11% | 40.11% | 9.89 pp | -35 | 8 | -4.38 |
| BTC Daily | xgb | XGBoost | 187 | 68 | 119 | 36.36% | 36.36% | 36.36% | 13.64 pp | -51 | 9 | -5.67 |
| BTC Daily | lstm | LSTM | 177 | 62 | 115 | 35.03% | 35.03% | 35.03% | 14.97 pp | -53 | 8 | -6.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 175 | 92 | 83 | 52.57% | 52.57% | 52.57% | 2.57 pp | 9 | 14 | 0.64 |
| BTC Market Hours | transformer | Transformer | 175 | 83 | 92 | 47.43% | 47.43% | 47.43% | 2.57 pp | -9 | 14 | -0.64 |
| BTC Market Hours | rf | RandomForest | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 14 | -0.93 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 14 | -1.07 |
| BTC Market Hours | xgb | XGBoost | 175 | 75 | 100 | 42.86% | 42.86% | 42.86% | 7.14 pp | -25 | 14 | -1.79 |
| BTC Market Hours | lstm | LSTM | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 14 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 15 | 0.07 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 15 | -0.33 |
| BTC Market Hours Daily | nn | NN | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 15 | -0.87 |
| BTC Market Hours Daily | rf | RandomForest | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 15 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 15 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 175 | 70 | 105 | 40.00% | 40.00% | 40.00% | 10.00 pp | -35 | 15 | -2.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Hourly | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 149 | 77 | 72 | 51.68% | 51.68% | 51.68% | 1.68 pp | 5 | 11 | 0.45 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 149 | 72 | 77 | 48.32% | 48.32% | 48.32% | 1.68 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 149 | 70 | 79 | 46.98% | 46.98% | 46.98% | 3.02 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 149 | 68 | 81 | 45.64% | 45.64% | 45.64% | 4.36 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 149 | 64 | 85 | 42.95% | 42.95% | 42.95% | 7.05 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 149 | 61 | 88 | 40.94% | 40.94% | 40.94% | 9.06 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 36 | 13 | 23 | 36.11% | 36.11% | 36.11% | 13.89 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | nn | NN | 36 | 12 | 24 | 33.33% | 33.33% | 33.33% | 16.67 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
