# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T18:36:20.181084+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 231 | 171 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 266 | 206 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 369 | 194 | 175 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 369 | 194 | 175 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 194 | 102 | 92 | 52.58% | 52.58% | 52.58% | 2.58 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 171 | 87 | 84 | 50.88% | 50.88% | 50.88% | 0.88 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 171 | 81 | 90 | 47.37% | 47.37% | 47.37% | 2.63 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | rf | RandomForest | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 16 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Daily | nn | NN | 196 | 88 | 108 | 44.90% | 44.90% | 44.90% | 5.10 pp | -20 | 9 | -2.22 |
| BTC Hourly | rf | RandomForest | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 8 | -3.12 |
| BTC Daily | transformer | Transformer | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| BTC Daily | rf | RandomForest | 196 | 75 | 121 | 38.27% | 38.27% | 38.27% | 11.73 pp | -46 | 9 | -5.11 |
| BTC Hourly | lstm | LSTM | 171 | 63 | 108 | 36.84% | 36.84% | 36.84% | 13.16 pp | -45 | 8 | -5.62 |
| BTC Daily | xgb | XGBoost | 206 | 73 | 133 | 35.44% | 35.44% | 35.44% | 14.56 pp | -60 | 10 | -6.00 |
| BTC Hourly | xgb | XGBoost | 171 | 61 | 110 | 35.67% | 35.67% | 35.67% | 14.33 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 196 | 67 | 129 | 34.18% | 34.18% | 34.18% | 15.82 pp | -62 | 9 | -6.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 171 | 87 | 84 | 50.88% | 50.88% | 50.88% | 0.88 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 171 | 81 | 90 | 47.37% | 47.37% | 47.37% | 2.63 pp | -9 | 8 | -1.12 |
| BTC Hourly | rf | RandomForest | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 8 | -2.88 |
| BTC Hourly | nn | NN | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 8 | -3.12 |
| BTC Hourly | lstm | LSTM | 171 | 63 | 108 | 36.84% | 36.84% | 36.84% | 13.16 pp | -45 | 8 | -5.62 |
| BTC Hourly | xgb | XGBoost | 171 | 61 | 110 | 35.67% | 35.67% | 35.67% | 14.33 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 94 | 102 | 47.96% | 47.96% | 47.96% | 2.04 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 196 | 88 | 108 | 44.90% | 44.90% | 44.90% | 5.10 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| BTC Daily | rf | RandomForest | 196 | 75 | 121 | 38.27% | 38.27% | 38.27% | 11.73 pp | -46 | 9 | -5.11 |
| BTC Daily | xgb | XGBoost | 206 | 73 | 133 | 35.44% | 35.44% | 35.44% | 14.56 pp | -60 | 10 | -6.00 |
| BTC Daily | lstm | LSTM | 196 | 67 | 129 | 34.18% | 34.18% | 34.18% | 15.82 pp | -62 | 9 | -6.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 194 | 102 | 92 | 52.58% | 52.58% | 52.58% | 2.58 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 16 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
