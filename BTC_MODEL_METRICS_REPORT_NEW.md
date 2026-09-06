# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T05:35:36.793202+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 238 | 178 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 274 | 214 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 384 | 202 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 384 | 202 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 175 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T21:00:00+00:00 | 176 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 202 | 104 | 98 | 51.49% | 51.49% | 51.49% | 1.49 pp | 6 | 17 | 0.35 |
| BTC Market Hours | nn | NN | 202 | 103 | 99 | 50.99% | 50.99% | 50.99% | 0.99 pp | 4 | 16 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 16 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 202 | 98 | 104 | 48.51% | 48.51% | 48.51% | 1.49 pp | -6 | 17 | -0.35 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 17 | -0.59 |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 202 | 95 | 107 | 47.03% | 47.03% | 47.03% | 2.97 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 9 | -1.11 |
| BTC Market Hours | rf | RandomForest | 202 | 92 | 110 | 45.54% | 45.54% | 45.54% | 4.46 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 17 | -1.53 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| BTC Hourly | transformer | Transformer | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 8 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 16 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 202 | 84 | 118 | 41.58% | 41.58% | 41.58% | 8.42 pp | -34 | 16 | -2.12 |
| BTC Daily | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 202 | 82 | 120 | 40.59% | 40.59% | 40.59% | 9.41 pp | -38 | 17 | -2.24 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 202 | 81 | 121 | 40.10% | 40.10% | 40.10% | 9.90 pp | -40 | 17 | -2.35 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 8 | -3.50 |
| BTC Daily | transformer | Transformer | 204 | 86 | 118 | 42.16% | 42.16% | 42.16% | 7.84 pp | -32 | 9 | -3.56 |
| BTC Daily | rf | RandomForest | 204 | 77 | 127 | 37.75% | 37.75% | 37.75% | 12.25 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 178 | 66 | 112 | 37.08% | 37.08% | 37.08% | 12.92 pp | -46 | 8 | -5.75 |
| BTC Hourly | lstm | LSTM | 178 | 65 | 113 | 36.52% | 36.52% | 36.52% | 13.48 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 214 | 76 | 138 | 35.51% | 35.51% | 35.51% | 14.49 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 204 | 68 | 136 | 33.33% | 33.33% | 33.33% | 16.67 pp | -68 | 9 | -7.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 178 | 82 | 96 | 46.07% | 46.07% | 46.07% | 3.93 pp | -14 | 8 | -1.75 |
| BTC Hourly | nn | NN | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 8 | -3.50 |
| BTC Hourly | rf | RandomForest | 178 | 75 | 103 | 42.13% | 42.13% | 42.13% | 7.87 pp | -28 | 8 | -3.50 |
| BTC Hourly | xgb | XGBoost | 178 | 66 | 112 | 37.08% | 37.08% | 37.08% | 12.92 pp | -46 | 8 | -5.75 |
| BTC Hourly | lstm | LSTM | 178 | 65 | 113 | 36.52% | 36.52% | 36.52% | 13.48 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 204 | 97 | 107 | 47.55% | 47.55% | 47.55% | 2.45 pp | -10 | 9 | -1.11 |
| BTC Daily | nn | NN | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 204 | 86 | 118 | 42.16% | 42.16% | 42.16% | 7.84 pp | -32 | 9 | -3.56 |
| BTC Daily | rf | RandomForest | 204 | 77 | 127 | 37.75% | 37.75% | 37.75% | 12.25 pp | -50 | 9 | -5.56 |
| BTC Daily | xgb | XGBoost | 214 | 76 | 138 | 35.51% | 35.51% | 35.51% | 14.49 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 204 | 68 | 136 | 33.33% | 33.33% | 33.33% | 16.67 pp | -68 | 9 | -7.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 202 | 103 | 99 | 50.99% | 50.99% | 50.99% | 0.99 pp | 4 | 16 | 0.25 |
| BTC Market Hours | transformer | Transformer | 202 | 99 | 103 | 49.01% | 49.01% | 49.01% | 0.99 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 202 | 95 | 107 | 47.03% | 47.03% | 47.03% | 2.97 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 202 | 92 | 110 | 45.54% | 45.54% | 45.54% | 4.46 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 202 | 87 | 115 | 43.07% | 43.07% | 43.07% | 6.93 pp | -28 | 16 | -1.75 |
| BTC Market Hours | lstm | LSTM | 202 | 84 | 118 | 41.58% | 41.58% | 41.58% | 8.42 pp | -34 | 16 | -2.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 202 | 104 | 98 | 51.49% | 51.49% | 51.49% | 1.49 pp | 6 | 17 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 202 | 98 | 104 | 48.51% | 48.51% | 48.51% | 1.49 pp | -6 | 17 | -0.35 |
| BTC Market Hours Daily | nn | NN | 202 | 96 | 106 | 47.52% | 47.52% | 47.52% | 2.48 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | rf | RandomForest | 202 | 88 | 114 | 43.56% | 43.56% | 43.56% | 6.44 pp | -26 | 17 | -1.53 |
| BTC Market Hours Daily | xgb | XGBoost | 202 | 82 | 120 | 40.59% | 40.59% | 40.59% | 9.41 pp | -38 | 17 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 202 | 81 | 121 | 40.10% | 40.10% | 40.10% | 9.90 pp | -40 | 17 | -2.35 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 175 | 86 | 89 | 49.14% | 49.14% | 49.14% | 0.86 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 175 | 85 | 90 | 48.57% | 48.57% | 48.57% | 1.43 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 175 | 80 | 95 | 45.71% | 45.71% | 45.71% | 4.29 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 175 | 78 | 97 | 44.57% | 44.57% | 44.57% | 5.43 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 50 | 21 | 29 | 42.00% | 42.00% | 42.00% | 8.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 51 | 24 | 27 | 47.06% | 47.06% | 47.06% | 2.94 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 51 | 21 | 30 | 41.18% | 41.18% | 41.18% | 8.82 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 51 | 20 | 31 | 39.22% | 39.22% | 39.22% | 10.78 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
