# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T09:17:24.044076+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 160 | 100 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 196 | 136 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 241 | 124 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 241 | 124 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 101 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 20:00:00+00:00 | 101 | 10 | 91 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| BTC Market Hours | nn | NN | 124 | 65 | 59 | 52.42% | 52.42% | 52.42% | 2.42 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 124 | 60 | 64 | 48.39% | 48.39% | 48.39% | 1.61 pp | -4 | 11 | -0.36 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 124 | 59 | 65 | 47.58% | 47.58% | 47.58% | 2.42 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| BTC Hourly | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 5 | -0.80 |
| BTC Hourly | transformer | Transformer | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 11 | -1.09 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 124 | 55 | 69 | 44.35% | 44.35% | 44.35% | 5.65 pp | -14 | 11 | -1.27 |
| BTC Daily | nn | NN | 126 | 59 | 67 | 46.83% | 46.83% | 46.83% | 3.17 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |
| BTC Market Hours | transformer | Transformer | 124 | 53 | 71 | 42.74% | 42.74% | 42.74% | 7.26 pp | -18 | 10 | -1.80 |
| BTC Daily | transformer | Transformer | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 124 | 48 | 76 | 38.71% | 38.71% | 38.71% | 11.29 pp | -28 | 11 | -2.55 |
| BTC Market Hours | xgb | XGBoost | 124 | 49 | 75 | 39.52% | 39.52% | 39.52% | 10.48 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 124 | 46 | 78 | 37.10% | 37.10% | 37.10% | 12.90 pp | -32 | 11 | -2.91 |
| BTC Market Hours | lstm | LSTM | 124 | 47 | 77 | 37.90% | 37.90% | 37.90% | 12.10 pp | -30 | 10 | -3.00 |
| BTC Daily | rf | RandomForest | 126 | 52 | 74 | 41.27% | 41.27% | 41.27% | 8.73 pp | -22 | 6 | -3.67 |
| BTC Hourly | rf | RandomForest | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 5 | -4.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 136 | 49 | 87 | 36.03% | 36.03% | 36.03% | 13.97 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 126 | 43 | 83 | 34.13% | 34.13% | 34.13% | 15.87 pp | -40 | 6 | -6.67 |
| BTC Hourly | xgb | XGBoost | 100 | 33 | 67 | 33.00% | 33.00% | 33.00% | 17.00 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 100 | 31 | 69 | 31.00% | 31.00% | 31.00% | 19.00 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 5 | -0.80 |
| BTC Hourly | transformer | Transformer | 100 | 48 | 52 | 48.00% | 48.00% | 48.00% | 2.00 pp | -4 | 5 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 100 | 33 | 67 | 33.00% | 33.00% | 33.00% | 17.00 pp | -34 | 5 | -6.80 |
| BTC Hourly | lstm | LSTM | 100 | 31 | 69 | 31.00% | 31.00% | 31.00% | 19.00 pp | -38 | 5 | -7.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 126 | 62 | 64 | 49.21% | 49.21% | 49.21% | 0.79 pp | -2 | 6 | -0.33 |
| BTC Daily | nn | NN | 126 | 59 | 67 | 46.83% | 46.83% | 46.83% | 3.17 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 126 | 57 | 69 | 45.24% | 45.24% | 45.24% | 4.76 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 126 | 52 | 74 | 41.27% | 41.27% | 41.27% | 8.73 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 136 | 49 | 87 | 36.03% | 36.03% | 36.03% | 13.97 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 126 | 43 | 83 | 34.13% | 34.13% | 34.13% | 15.87 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 124 | 65 | 59 | 52.42% | 52.42% | 52.42% | 2.42 pp | 6 | 10 | 0.60 |
| BTC Market Hours | rf | RandomForest | 124 | 59 | 65 | 47.58% | 47.58% | 47.58% | 2.42 pp | -6 | 10 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 10 | -1.20 |
| BTC Market Hours | transformer | Transformer | 124 | 53 | 71 | 42.74% | 42.74% | 42.74% | 7.26 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 124 | 49 | 75 | 39.52% | 39.52% | 39.52% | 10.48 pp | -26 | 10 | -2.60 |
| BTC Market Hours | lstm | LSTM | 124 | 47 | 77 | 37.90% | 37.90% | 37.90% | 12.10 pp | -30 | 10 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 124 | 60 | 64 | 48.39% | 48.39% | 48.39% | 1.61 pp | -4 | 11 | -0.36 |
| BTC Market Hours Daily | rf | RandomForest | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 124 | 56 | 68 | 45.16% | 45.16% | 45.16% | 4.84 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | nn | NN | 124 | 55 | 69 | 44.35% | 44.35% | 44.35% | 5.65 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 124 | 48 | 76 | 38.71% | 38.71% | 38.71% | 11.29 pp | -28 | 11 | -2.55 |
| BTC Market Hours Daily | lstm | LSTM | 124 | 46 | 78 | 37.10% | 37.10% | 37.10% | 12.90 pp | -32 | 11 | -2.91 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 7 | 3 | 70.00% | 70.00% | 70.00% | 20.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
