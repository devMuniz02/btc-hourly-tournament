# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-04T02:12:32.156904+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 203 | 143 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 239 | 179 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 323 | 167 | 156 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 00:00:00+00:00 | 323 | 167 | 156 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 141 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-28T16:00:00+00:00 | 142 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 6 | 1.17 |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 167 | 87 | 80 | 52.10% | 52.10% | 52.10% | 2.10 pp | 7 | 13 | 0.54 |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| BTC Market Hours Daily | transformer | Transformer | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 167 | 80 | 87 | 47.90% | 47.90% | 47.90% | 2.10 pp | -7 | 14 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| BTC Hourly | transformer | Transformer | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 6 | -0.83 |
| BTC Market Hours | rf | RandomForest | 167 | 78 | 89 | 46.71% | 46.71% | 46.71% | 3.29 pp | -11 | 13 | -0.85 |
| BTC Market Hours Daily | nn | NN | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 14 | -0.93 |
| Consolidated Hourly | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| BTC Market Hours | transformer | Transformer | 167 | 76 | 91 | 45.51% | 45.51% | 45.51% | 4.49 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 14 | -1.36 |
| Consolidated Hourly | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 8 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 167 | 70 | 97 | 41.92% | 41.92% | 41.92% | 8.08 pp | -27 | 14 | -1.93 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |
| BTC Daily | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 8 | -2.12 |
| BTC Market Hours | lstm | LSTM | 167 | 69 | 98 | 41.32% | 41.32% | 41.32% | 8.68 pp | -29 | 13 | -2.23 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| BTC Daily | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | lstm | LSTM | 167 | 65 | 102 | 38.92% | 38.92% | 38.92% | 11.08 pp | -37 | 14 | -2.64 |
| BTC Hourly | nn | NN | 143 | 62 | 81 | 43.36% | 43.36% | 43.36% | 6.64 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 6 | -3.83 |
| BTC Daily | rf | RandomForest | 169 | 69 | 100 | 40.83% | 40.83% | 40.83% | 9.17 pp | -31 | 8 | -3.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 9 | 24 | 27.27% | 27.27% | 27.27% | 22.73 pp | -15 | 3 | -5.00 |
| BTC Daily | xgb | XGBoost | 179 | 66 | 113 | 36.87% | 36.87% | 36.87% | 13.13 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 169 | 60 | 109 | 35.50% | 35.50% | 35.50% | 14.50 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 143 | 52 | 91 | 36.36% | 36.36% | 36.36% | 13.64 pp | -39 | 6 | -6.50 |
| BTC Hourly | lstm | LSTM | 143 | 51 | 92 | 35.66% | 35.66% | 35.66% | 14.34 pp | -41 | 6 | -6.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 143 | 75 | 68 | 52.45% | 52.45% | 52.45% | 2.45 pp | 7 | 6 | 1.17 |
| BTC Hourly | transformer | Transformer | 143 | 69 | 74 | 48.25% | 48.25% | 48.25% | 1.75 pp | -5 | 6 | -0.83 |
| BTC Hourly | nn | NN | 143 | 62 | 81 | 43.36% | 43.36% | 43.36% | 6.64 pp | -19 | 6 | -3.17 |
| BTC Hourly | rf | RandomForest | 143 | 60 | 83 | 41.96% | 41.96% | 41.96% | 8.04 pp | -23 | 6 | -3.83 |
| BTC Hourly | xgb | XGBoost | 143 | 52 | 91 | 36.36% | 36.36% | 36.36% | 13.64 pp | -39 | 6 | -6.50 |
| BTC Hourly | lstm | LSTM | 143 | 51 | 92 | 35.66% | 35.66% | 35.66% | 14.34 pp | -41 | 6 | -6.83 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 169 | 78 | 91 | 46.15% | 46.15% | 46.15% | 3.85 pp | -13 | 8 | -1.62 |
| BTC Daily | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 169 | 74 | 95 | 43.79% | 43.79% | 43.79% | 6.21 pp | -21 | 8 | -2.62 |
| BTC Daily | rf | RandomForest | 169 | 69 | 100 | 40.83% | 40.83% | 40.83% | 9.17 pp | -31 | 8 | -3.88 |
| BTC Daily | xgb | XGBoost | 179 | 66 | 113 | 36.87% | 36.87% | 36.87% | 13.13 pp | -47 | 9 | -5.22 |
| BTC Daily | lstm | LSTM | 169 | 60 | 109 | 35.50% | 35.50% | 35.50% | 14.50 pp | -49 | 8 | -6.12 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 167 | 87 | 80 | 52.10% | 52.10% | 52.10% | 2.10 pp | 7 | 13 | 0.54 |
| BTC Market Hours | rf | RandomForest | 167 | 78 | 89 | 46.71% | 46.71% | 46.71% | 3.29 pp | -11 | 13 | -0.85 |
| BTC Market Hours | transformer | Transformer | 167 | 76 | 91 | 45.51% | 45.51% | 45.51% | 4.49 pp | -15 | 13 | -1.15 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 13 | -1.92 |
| BTC Market Hours | lstm | LSTM | 167 | 69 | 98 | 41.32% | 41.32% | 41.32% | 8.68 pp | -29 | 13 | -2.23 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 167 | 80 | 87 | 47.90% | 47.90% | 47.90% | 2.10 pp | -7 | 14 | -0.50 |
| BTC Market Hours Daily | nn | NN | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 14 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 14 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 167 | 70 | 97 | 41.92% | 41.92% | 41.92% | 8.08 pp | -27 | 14 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 167 | 65 | 102 | 38.92% | 38.92% | 38.92% | 11.08 pp | -37 | 14 | -2.64 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Hourly | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Hourly | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Hourly | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 141 | 72 | 69 | 51.06% | 51.06% | 51.06% | 1.06 pp | 3 | 11 | 0.27 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 141 | 69 | 72 | 48.94% | 48.94% | 48.94% | 1.06 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 141 | 66 | 75 | 46.81% | 46.81% | 46.81% | 3.19 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 141 | 65 | 76 | 46.10% | 46.10% | 46.10% | 3.90 pp | -11 | 11 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 141 | 63 | 78 | 44.68% | 44.68% | 44.68% | 5.32 pp | -15 | 11 | -1.36 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 141 | 59 | 82 | 41.84% | 41.84% | 41.84% | 8.16 pp | -23 | 11 | -2.09 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | lstm | LSTM | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | rf | RandomForest | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 32 | 13 | 19 | 40.62% | 40.62% | 40.62% | 9.38 pp | -6 | 3 | -2.00 |
| Consolidated Market Hours | nn | NN | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 32 | 9 | 23 | 28.12% | 28.12% | 28.12% | 21.88 pp | -14 | 3 | -4.67 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 33 | 17 | 16 | 51.52% | 51.52% | 51.52% | 1.52 pp | 1 | 3 | 0.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 3 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 33 | 13 | 20 | 39.39% | 39.39% | 39.39% | 10.61 pp | -7 | 3 | -2.33 |
| Consolidated Market Hours Daily | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 33 | 9 | 24 | 27.27% | 27.27% | 27.27% | 22.73 pp | -15 | 3 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
