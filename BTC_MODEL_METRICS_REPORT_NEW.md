# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T19:28:15.648541+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 215 | 155 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 250 | 190 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 341 | 178 | 163 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 18:00:00+00:00 | 341 | 178 | 163 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 153 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 22:00:00+00:00 | 153 | 38 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 7 | 1.00 |
| BTC Market Hours | nn | NN | 178 | 93 | 85 | 52.25% | 52.25% | 52.25% | 2.25 pp | 8 | 14 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 15 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| BTC Market Hours | transformer | Transformer | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 14 | -0.43 |
| BTC Market Hours Daily | nn | NN | 178 | 84 | 94 | 47.19% | 47.19% | 47.19% | 2.81 pp | -10 | 15 | -0.67 |
| BTC Hourly | transformer | Transformer | 155 | 75 | 80 | 48.39% | 48.39% | 48.39% | 1.61 pp | -5 | 7 | -0.71 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| BTC Market Hours | rf | RandomForest | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 178 | 80 | 98 | 44.94% | 44.94% | 44.94% | 5.06 pp | -18 | 14 | -1.29 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 15 | -1.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 180 | 84 | 96 | 46.67% | 46.67% | 46.67% | 3.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| BTC Market Hours | xgb | XGBoost | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 14 | -1.71 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| BTC Market Hours Daily | xgb | XGBoost | 178 | 74 | 104 | 41.57% | 41.57% | 41.57% | 8.43 pp | -30 | 15 | -2.00 |
| BTC Market Hours | lstm | LSTM | 178 | 73 | 105 | 41.01% | 41.01% | 41.01% | 8.99 pp | -32 | 14 | -2.29 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 178 | 70 | 108 | 39.33% | 39.33% | 39.33% | 10.67 pp | -38 | 15 | -2.53 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| BTC Daily | nn | NN | 180 | 79 | 101 | 43.89% | 43.89% | 43.89% | 6.11 pp | -22 | 8 | -2.75 |
| BTC Daily | transformer | Transformer | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 7 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| BTC Hourly | rf | RandomForest | 155 | 65 | 90 | 41.94% | 41.94% | 41.94% | 8.06 pp | -25 | 7 | -3.57 |
| BTC Daily | rf | RandomForest | 180 | 72 | 108 | 40.00% | 40.00% | 40.00% | 10.00 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 190 | 68 | 122 | 35.79% | 35.79% | 35.79% | 14.21 pp | -54 | 9 | -6.00 |
| BTC Hourly | lstm | LSTM | 155 | 56 | 99 | 36.13% | 36.13% | 36.13% | 13.87 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 155 | 56 | 99 | 36.13% | 36.13% | 36.13% | 13.87 pp | -43 | 7 | -6.14 |
| BTC Daily | lstm | LSTM | 180 | 63 | 117 | 35.00% | 35.00% | 35.00% | 15.00 pp | -54 | 8 | -6.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 7 | 1.00 |
| BTC Hourly | transformer | Transformer | 155 | 75 | 80 | 48.39% | 48.39% | 48.39% | 1.61 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 155 | 67 | 88 | 43.23% | 43.23% | 43.23% | 6.77 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 155 | 65 | 90 | 41.94% | 41.94% | 41.94% | 8.06 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 155 | 56 | 99 | 36.13% | 36.13% | 36.13% | 13.87 pp | -43 | 7 | -6.14 |
| BTC Hourly | xgb | XGBoost | 155 | 56 | 99 | 36.13% | 36.13% | 36.13% | 13.87 pp | -43 | 7 | -6.14 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 180 | 84 | 96 | 46.67% | 46.67% | 46.67% | 3.33 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 180 | 79 | 101 | 43.89% | 43.89% | 43.89% | 6.11 pp | -22 | 8 | -2.75 |
| BTC Daily | transformer | Transformer | 180 | 78 | 102 | 43.33% | 43.33% | 43.33% | 6.67 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 180 | 72 | 108 | 40.00% | 40.00% | 40.00% | 10.00 pp | -36 | 8 | -4.50 |
| BTC Daily | xgb | XGBoost | 190 | 68 | 122 | 35.79% | 35.79% | 35.79% | 14.21 pp | -54 | 9 | -6.00 |
| BTC Daily | lstm | LSTM | 180 | 63 | 117 | 35.00% | 35.00% | 35.00% | 15.00 pp | -54 | 8 | -6.75 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 178 | 93 | 85 | 52.25% | 52.25% | 52.25% | 2.25 pp | 8 | 14 | 0.57 |
| BTC Market Hours | transformer | Transformer | 178 | 86 | 92 | 48.31% | 48.31% | 48.31% | 1.69 pp | -6 | 14 | -0.43 |
| BTC Market Hours | rf | RandomForest | 178 | 81 | 97 | 45.51% | 45.51% | 45.51% | 4.49 pp | -16 | 14 | -1.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 178 | 80 | 98 | 44.94% | 44.94% | 44.94% | 5.06 pp | -18 | 14 | -1.29 |
| BTC Market Hours | xgb | XGBoost | 178 | 77 | 101 | 43.26% | 43.26% | 43.26% | 6.74 pp | -24 | 14 | -1.71 |
| BTC Market Hours | lstm | LSTM | 178 | 73 | 105 | 41.01% | 41.01% | 41.01% | 8.99 pp | -32 | 14 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 178 | 91 | 87 | 51.12% | 51.12% | 51.12% | 1.12 pp | 4 | 15 | 0.27 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 178 | 87 | 91 | 48.88% | 48.88% | 48.88% | 1.12 pp | -4 | 15 | -0.27 |
| BTC Market Hours Daily | nn | NN | 178 | 84 | 94 | 47.19% | 47.19% | 47.19% | 2.81 pp | -10 | 15 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 178 | 78 | 100 | 43.82% | 43.82% | 43.82% | 6.18 pp | -22 | 15 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 178 | 74 | 104 | 41.57% | 41.57% | 41.57% | 8.43 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 178 | 70 | 108 | 39.33% | 39.33% | 39.33% | 10.67 pp | -38 | 15 | -2.53 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Hourly | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Hourly | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 153 | 77 | 76 | 50.33% | 50.33% | 50.33% | 0.33 pp | 1 | 11 | 0.09 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 153 | 75 | 78 | 49.02% | 49.02% | 49.02% | 0.98 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 153 | 71 | 82 | 46.41% | 46.41% | 46.41% | 3.59 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 153 | 68 | 85 | 44.44% | 44.44% | 44.44% | 5.56 pp | -17 | 11 | -1.55 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 153 | 67 | 86 | 43.79% | 43.79% | 43.79% | 6.21 pp | -19 | 11 | -1.73 |
| Consolidated Daily/Hourly Refresh | nn | NN | 153 | 63 | 90 | 41.18% | 41.18% | 41.18% | 8.82 pp | -27 | 11 | -2.45 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 38 | 14 | 24 | 36.84% | 36.84% | 36.84% | 13.16 pp | -10 | 3 | -3.33 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
