# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T10:49:23.490798+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 144 | 84 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 180 | 120 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 212 | 108 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 212 | 108 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 87 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 108 | 57 | 51 | 52.78% | 52.78% | 52.78% | 2.78 pp | 6 | 9 | 0.67 |
| BTC Hourly | transformer | Transformer | 84 | 43 | 41 | 51.19% | 51.19% | 51.19% | 1.19 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | xgb | XGBoost | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 108 | 54 | 54 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Hourly | nn | NN | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 5 | -0.40 |
| BTC Market Hours | rf | RandomForest | 108 | 52 | 56 | 48.15% | 48.15% | 48.15% | 1.85 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | lstm | LSTM | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 9 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 108 | 51 | 57 | 47.22% | 47.22% | 47.22% | 2.78 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | nn | NN | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 9 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 9 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 108 | 49 | 59 | 45.37% | 45.37% | 45.37% | 4.63 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 108 | 49 | 59 | 45.37% | 45.37% | 45.37% | 4.63 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Daily | nn | NN | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | nn | NN | 108 | 47 | 61 | 43.52% | 43.52% | 43.52% | 6.48 pp | -14 | 10 | -1.40 |
| BTC Market Hours | transformer | Transformer | 108 | 45 | 63 | 41.67% | 41.67% | 41.67% | 8.33 pp | -18 | 9 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 5 | -2.40 |
| BTC Market Hours | lstm | LSTM | 108 | 43 | 65 | 39.81% | 39.81% | 39.81% | 10.19 pp | -22 | 9 | -2.44 |
| BTC Market Hours | xgb | XGBoost | 108 | 43 | 65 | 39.81% | 39.81% | 39.81% | 10.19 pp | -22 | 9 | -2.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 4 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 108 | 41 | 67 | 37.96% | 37.96% | 37.96% | 12.04 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | xgb | XGBoost | 108 | 41 | 67 | 37.96% | 37.96% | 37.96% | 12.04 pp | -26 | 10 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 110 | 45 | 65 | 40.91% | 40.91% | 40.91% | 9.09 pp | -20 | 5 | -4.00 |
| BTC Daily | xgb | XGBoost | 120 | 43 | 77 | 35.83% | 35.83% | 35.83% | 14.17 pp | -34 | 6 | -5.67 |
| BTC Daily | lstm | LSTM | 110 | 39 | 71 | 35.45% | 35.45% | 35.45% | 14.55 pp | -32 | 5 | -6.40 |
| BTC Hourly | xgb | XGBoost | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 4 | -6.50 |
| BTC Hourly | lstm | LSTM | 84 | 28 | 56 | 33.33% | 33.33% | 33.33% | 16.67 pp | -28 | 4 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 84 | 43 | 41 | 51.19% | 51.19% | 51.19% | 1.19 pp | 2 | 4 | 0.50 |
| BTC Hourly | nn | NN | 84 | 42 | 42 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 84 | 37 | 47 | 44.05% | 44.05% | 44.05% | 5.95 pp | -10 | 4 | -2.50 |
| BTC Hourly | rf | RandomForest | 84 | 35 | 49 | 41.67% | 41.67% | 41.67% | 8.33 pp | -14 | 4 | -3.50 |
| BTC Hourly | xgb | XGBoost | 84 | 29 | 55 | 34.52% | 34.52% | 34.52% | 15.48 pp | -26 | 4 | -6.50 |
| BTC Hourly | lstm | LSTM | 84 | 28 | 56 | 33.33% | 33.33% | 33.33% | 16.67 pp | -28 | 4 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 110 | 54 | 56 | 49.09% | 49.09% | 49.09% | 0.91 pp | -2 | 5 | -0.40 |
| BTC Daily | nn | NN | 110 | 52 | 58 | 47.27% | 47.27% | 47.27% | 2.73 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 5 | -2.40 |
| BTC Daily | rf | RandomForest | 110 | 45 | 65 | 40.91% | 40.91% | 40.91% | 9.09 pp | -20 | 5 | -4.00 |
| BTC Daily | xgb | XGBoost | 120 | 43 | 77 | 35.83% | 35.83% | 35.83% | 14.17 pp | -34 | 6 | -5.67 |
| BTC Daily | lstm | LSTM | 110 | 39 | 71 | 35.45% | 35.45% | 35.45% | 14.55 pp | -32 | 5 | -6.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 108 | 57 | 51 | 52.78% | 52.78% | 52.78% | 2.78 pp | 6 | 9 | 0.67 |
| BTC Market Hours | rf | RandomForest | 108 | 52 | 56 | 48.15% | 48.15% | 48.15% | 1.85 pp | -4 | 9 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 108 | 51 | 57 | 47.22% | 47.22% | 47.22% | 2.78 pp | -6 | 9 | -0.67 |
| BTC Market Hours | transformer | Transformer | 108 | 45 | 63 | 41.67% | 41.67% | 41.67% | 8.33 pp | -18 | 9 | -2.00 |
| BTC Market Hours | lstm | LSTM | 108 | 43 | 65 | 39.81% | 39.81% | 39.81% | 10.19 pp | -22 | 9 | -2.44 |
| BTC Market Hours | xgb | XGBoost | 108 | 43 | 65 | 39.81% | 39.81% | 39.81% | 10.19 pp | -22 | 9 | -2.44 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 108 | 54 | 54 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 10 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 108 | 49 | 59 | 45.37% | 45.37% | 45.37% | 4.63 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 108 | 49 | 59 | 45.37% | 45.37% | 45.37% | 4.63 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | nn | NN | 108 | 47 | 61 | 43.52% | 43.52% | 43.52% | 6.48 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 108 | 41 | 67 | 37.96% | 37.96% | 37.96% | 12.04 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | xgb | XGBoost | 108 | 41 | 67 | 37.96% | 37.96% | 37.96% | 12.04 pp | -26 | 10 | -2.60 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | xgb | XGBoost | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | nn | NN | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 9 | -1.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 86 | 45 | 41 | 52.33% | 52.33% | 52.33% | 2.33 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 86 | 41 | 45 | 47.67% | 47.67% | 47.67% | 2.33 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 9 | -0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 86 | 37 | 49 | 43.02% | 43.02% | 43.02% | 6.98 pp | -12 | 9 | -1.33 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
