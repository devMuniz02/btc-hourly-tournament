# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T08:09:46.537264+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 256 | 196 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 292 | 232 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 415 | 220 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 415 | 220 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 220 | 114 | 106 | 51.82% | 51.82% | 51.82% | 1.82 pp | 8 | 17 | 0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 196 | 99 | 97 | 50.51% | 50.51% | 50.51% | 0.51 pp | 2 | 9 | 0.22 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 220 | 107 | 113 | 48.64% | 48.64% | 48.64% | 1.36 pp | -6 | 18 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | transformer | Transformer | 220 | 106 | 114 | 48.18% | 48.18% | 48.18% | 1.82 pp | -8 | 18 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 220 | 105 | 115 | 47.73% | 47.73% | 47.73% | 2.27 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 220 | 105 | 115 | 47.73% | 47.73% | 47.73% | 2.27 pp | -10 | 17 | -0.59 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 220 | 104 | 116 | 47.27% | 47.27% | 47.27% | 2.73 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 220 | 101 | 119 | 45.91% | 45.91% | 45.91% | 4.09 pp | -18 | 18 | -1.00 |
| BTC Market Hours | transformer | Transformer | 220 | 101 | 119 | 45.91% | 45.91% | 45.91% | 4.09 pp | -18 | 17 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| BTC Daily | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 220 | 94 | 126 | 42.73% | 42.73% | 42.73% | 7.27 pp | -32 | 17 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 220 | 93 | 127 | 42.27% | 42.27% | 42.27% | 7.73 pp | -34 | 18 | -1.89 |
| BTC Daily | nn | NN | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 10 | -2.20 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 196 | 88 | 108 | 44.90% | 44.90% | 44.90% | 5.10 pp | -20 | 9 | -2.22 |
| Consolidated Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| BTC Market Hours Daily | lstm | LSTM | 220 | 76 | 144 | 34.55% | 34.55% | 34.55% | 15.45 pp | -68 | 18 | -3.78 |
| BTC Market Hours | lstm | LSTM | 220 | 77 | 143 | 35.00% | 35.00% | 35.00% | 15.00 pp | -66 | 17 | -3.88 |
| BTC Daily | transformer | Transformer | 222 | 91 | 131 | 40.99% | 40.99% | 40.99% | 9.01 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 222 | 86 | 136 | 38.74% | 38.74% | 38.74% | 11.26 pp | -50 | 10 | -5.00 |
| BTC Hourly | lstm | LSTM | 196 | 73 | 123 | 37.24% | 37.24% | 37.24% | 12.76 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 196 | 71 | 125 | 36.22% | 36.22% | 36.22% | 13.78 pp | -54 | 9 | -6.00 |
| BTC Daily | xgb | XGBoost | 232 | 82 | 150 | 35.34% | 35.34% | 35.34% | 14.66 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 222 | 75 | 147 | 33.78% | 33.78% | 33.78% | 16.22 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 196 | 99 | 97 | 50.51% | 50.51% | 50.51% | 0.51 pp | 2 | 9 | 0.22 |
| BTC Hourly | transformer | Transformer | 196 | 88 | 108 | 44.90% | 44.90% | 44.90% | 5.10 pp | -20 | 9 | -2.22 |
| BTC Hourly | nn | NN | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 196 | 83 | 113 | 42.35% | 42.35% | 42.35% | 7.65 pp | -30 | 9 | -3.33 |
| BTC Hourly | lstm | LSTM | 196 | 73 | 123 | 37.24% | 37.24% | 37.24% | 12.76 pp | -50 | 9 | -5.56 |
| BTC Hourly | xgb | XGBoost | 196 | 71 | 125 | 36.22% | 36.22% | 36.22% | 13.78 pp | -54 | 9 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 222 | 103 | 119 | 46.40% | 46.40% | 46.40% | 3.60 pp | -16 | 10 | -1.60 |
| BTC Daily | nn | NN | 222 | 100 | 122 | 45.05% | 45.05% | 45.05% | 4.95 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 222 | 91 | 131 | 40.99% | 40.99% | 40.99% | 9.01 pp | -40 | 10 | -4.00 |
| BTC Daily | rf | RandomForest | 222 | 86 | 136 | 38.74% | 38.74% | 38.74% | 11.26 pp | -50 | 10 | -5.00 |
| BTC Daily | xgb | XGBoost | 232 | 82 | 150 | 35.34% | 35.34% | 35.34% | 14.66 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 222 | 75 | 147 | 33.78% | 33.78% | 33.78% | 16.22 pp | -72 | 10 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 220 | 114 | 106 | 51.82% | 51.82% | 51.82% | 1.82 pp | 8 | 17 | 0.47 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 220 | 105 | 115 | 47.73% | 47.73% | 47.73% | 2.27 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 220 | 105 | 115 | 47.73% | 47.73% | 47.73% | 2.27 pp | -10 | 17 | -0.59 |
| BTC Market Hours | transformer | Transformer | 220 | 101 | 119 | 45.91% | 45.91% | 45.91% | 4.09 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 220 | 94 | 126 | 42.73% | 42.73% | 42.73% | 7.27 pp | -32 | 17 | -1.88 |
| BTC Market Hours | lstm | LSTM | 220 | 77 | 143 | 35.00% | 35.00% | 35.00% | 15.00 pp | -66 | 17 | -3.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 220 | 107 | 113 | 48.64% | 48.64% | 48.64% | 1.36 pp | -6 | 18 | -0.33 |
| BTC Market Hours Daily | transformer | Transformer | 220 | 106 | 114 | 48.18% | 48.18% | 48.18% | 1.82 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 220 | 104 | 116 | 47.27% | 47.27% | 47.27% | 2.73 pp | -12 | 18 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 220 | 101 | 119 | 45.91% | 45.91% | 45.91% | 4.09 pp | -18 | 18 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 220 | 93 | 127 | 42.27% | 42.27% | 42.27% | 7.73 pp | -34 | 18 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 220 | 76 | 144 | 34.55% | 34.55% | 34.55% | 15.45 pp | -68 | 18 | -3.78 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
