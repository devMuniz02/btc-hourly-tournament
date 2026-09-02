# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-02T06:39:21.105132+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 174 | 114 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 210 | 150 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 268 | 138 | 130 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-02 00:00:00+00:00 | 268 | 138 | 130 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 113 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 14:00:00+00:00 | 113 | 17 | 96 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 138 | 72 | 66 | 52.17% | 52.17% | 52.17% | 2.17 pp | 6 | 11 | 0.55 |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 12 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 140 | 68 | 72 | 48.57% | 48.57% | 48.57% | 1.43 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | transformer | Transformer | 138 | 65 | 73 | 47.10% | 47.10% | 47.10% | 2.90 pp | -8 | 12 | -0.67 |
| BTC Market Hours | rf | RandomForest | 138 | 65 | 73 | 47.10% | 47.10% | 47.10% | 2.90 pp | -8 | 11 | -0.73 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 138 | 63 | 75 | 45.65% | 45.65% | 45.65% | 4.35 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 11 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |
| BTC Daily | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 7 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 138 | 57 | 81 | 41.30% | 41.30% | 41.30% | 8.70 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| BTC Daily | transformer | Transformer | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | lstm | LSTM | 138 | 52 | 86 | 37.68% | 37.68% | 37.68% | 12.32 pp | -34 | 12 | -2.83 |
| BTC Market Hours | lstm | LSTM | 138 | 53 | 85 | 38.41% | 38.41% | 38.41% | 11.59 pp | -32 | 11 | -2.91 |
| BTC Daily | rf | RandomForest | 140 | 58 | 82 | 41.43% | 41.43% | 41.43% | 8.57 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 5 | -3.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 150 | 54 | 96 | 36.00% | 36.00% | 36.00% | 14.00 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 140 | 49 | 91 | 35.00% | 35.00% | 35.00% | 15.00 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 114 | 41 | 73 | 35.96% | 35.96% | 35.96% | 14.04 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 114 | 37 | 77 | 32.46% | 32.46% | 32.46% | 17.54 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 114 | 56 | 58 | 49.12% | 49.12% | 49.12% | 0.88 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | transformer | Transformer | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 114 | 41 | 73 | 35.96% | 35.96% | 35.96% | 14.04 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 114 | 37 | 77 | 32.46% | 32.46% | 32.46% | 17.54 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 140 | 68 | 72 | 48.57% | 48.57% | 48.57% | 1.43 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 140 | 63 | 77 | 45.00% | 45.00% | 45.00% | 5.00 pp | -14 | 7 | -2.00 |
| BTC Daily | transformer | Transformer | 140 | 61 | 79 | 43.57% | 43.57% | 43.57% | 6.43 pp | -18 | 7 | -2.57 |
| BTC Daily | rf | RandomForest | 140 | 58 | 82 | 41.43% | 41.43% | 41.43% | 8.57 pp | -24 | 7 | -3.43 |
| BTC Daily | xgb | XGBoost | 150 | 54 | 96 | 36.00% | 36.00% | 36.00% | 14.00 pp | -42 | 8 | -5.25 |
| BTC Daily | lstm | LSTM | 140 | 49 | 91 | 35.00% | 35.00% | 35.00% | 15.00 pp | -42 | 7 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 138 | 72 | 66 | 52.17% | 52.17% | 52.17% | 2.17 pp | 6 | 11 | 0.55 |
| BTC Market Hours | rf | RandomForest | 138 | 65 | 73 | 47.10% | 47.10% | 47.10% | 2.90 pp | -8 | 11 | -0.73 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 138 | 63 | 75 | 45.65% | 45.65% | 45.65% | 4.35 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 138 | 61 | 77 | 44.20% | 44.20% | 44.20% | 5.80 pp | -16 | 11 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 138 | 58 | 80 | 42.03% | 42.03% | 42.03% | 7.97 pp | -22 | 11 | -2.00 |
| BTC Market Hours | lstm | LSTM | 138 | 53 | 85 | 38.41% | 38.41% | 38.41% | 11.59 pp | -32 | 11 | -2.91 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 138 | 67 | 71 | 48.55% | 48.55% | 48.55% | 1.45 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 138 | 65 | 73 | 47.10% | 47.10% | 47.10% | 2.90 pp | -8 | 12 | -0.67 |
| BTC Market Hours Daily | nn | NN | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 138 | 62 | 76 | 44.93% | 44.93% | 44.93% | 5.07 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 138 | 57 | 81 | 41.30% | 41.30% | 41.30% | 8.70 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 138 | 52 | 86 | 37.68% | 37.68% | 37.68% | 12.32 pp | -34 | 12 | -2.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 113 | 59 | 54 | 52.21% | 52.21% | 52.21% | 2.21 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 113 | 54 | 59 | 47.79% | 47.79% | 47.79% | 2.21 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 113 | 51 | 62 | 45.13% | 45.13% | 45.13% | 4.87 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 113 | 47 | 66 | 41.59% | 41.59% | 41.59% | 8.41 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 17 | 7 | 10 | 41.18% | 41.18% | 41.18% | 8.82 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 4 | 13 | 23.53% | 23.53% | 23.53% | 26.47 pp | -9 | 2 | -4.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
