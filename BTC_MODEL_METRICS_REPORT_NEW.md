# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T04:08:14.530786+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 237 | 177 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 273 | 213 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 383 | 201 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 383 | 201 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 173 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 173 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 173 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 174 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 201 | 104 | 97 | 51.74% | 51.74% | 51.74% | 1.74 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 8 | 0.38 |
| BTC Market Hours | nn | NN | 201 | 102 | 99 | 50.75% | 50.75% | 50.75% | 0.75 pp | 3 | 16 | 0.19 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 201 | 98 | 103 | 48.76% | 48.76% | 48.76% | 1.24 pp | -5 | 16 | -0.31 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 17 | -0.41 |
| Consolidated Hourly | rf | RandomForest | 173 | 84 | 89 | 48.55% | 48.55% | 48.55% | 1.45 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 84 | 89 | 48.55% | 48.55% | 48.55% | 1.45 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | nn | NN | 201 | 96 | 105 | 47.76% | 47.76% | 47.76% | 2.24 pp | -9 | 17 | -0.53 |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 201 | 94 | 107 | 46.77% | 46.77% | 46.77% | 3.23 pp | -13 | 16 | -0.81 |
| BTC Daily | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 173 | 78 | 95 | 45.09% | 45.09% | 45.09% | 4.91 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 78 | 95 | 45.09% | 45.09% | 45.09% | 4.91 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 17 | -1.47 |
| Consolidated Hourly | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 12 | -1.58 |
| BTC Hourly | transformer | Transformer | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 201 | 86 | 115 | 42.79% | 42.79% | 42.79% | 7.21 pp | -29 | 16 | -1.81 |
| BTC Market Hours | lstm | LSTM | 201 | 84 | 117 | 41.79% | 41.79% | 41.79% | 8.21 pp | -33 | 16 | -2.06 |
| Consolidated Hourly | transformer | Transformer | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 12 | -2.08 |
| BTC Daily | nn | NN | 203 | 92 | 111 | 45.32% | 45.32% | 45.32% | 4.68 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 201 | 82 | 119 | 40.80% | 40.80% | 40.80% | 9.20 pp | -37 | 17 | -2.18 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 201 | 81 | 120 | 40.30% | 40.30% | 40.30% | 9.70 pp | -39 | 17 | -2.29 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Hourly | rf | RandomForest | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Hourly | nn | NN | 177 | 74 | 103 | 41.81% | 41.81% | 41.81% | 8.19 pp | -29 | 8 | -3.62 |
| BTC Daily | rf | RandomForest | 203 | 77 | 126 | 37.93% | 37.93% | 37.93% | 12.07 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 8 | -5.88 |
| BTC Daily | xgb | XGBoost | 213 | 76 | 137 | 35.68% | 35.68% | 35.68% | 14.32 pp | -61 | 10 | -6.10 |
| BTC Hourly | lstm | LSTM | 177 | 64 | 113 | 36.16% | 36.16% | 36.16% | 13.84 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 203 | 68 | 135 | 33.50% | 33.50% | 33.50% | 16.50 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 177 | 90 | 87 | 50.85% | 50.85% | 50.85% | 0.85 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 177 | 82 | 95 | 46.33% | 46.33% | 46.33% | 3.67 pp | -13 | 8 | -1.62 |
| BTC Hourly | rf | RandomForest | 177 | 75 | 102 | 42.37% | 42.37% | 42.37% | 7.63 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 177 | 74 | 103 | 41.81% | 41.81% | 41.81% | 8.19 pp | -29 | 8 | -3.62 |
| BTC Hourly | xgb | XGBoost | 177 | 65 | 112 | 36.72% | 36.72% | 36.72% | 13.28 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 177 | 64 | 113 | 36.16% | 36.16% | 36.16% | 13.84 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 203 | 97 | 106 | 47.78% | 47.78% | 47.78% | 2.22 pp | -9 | 9 | -1.00 |
| BTC Daily | nn | NN | 203 | 92 | 111 | 45.32% | 45.32% | 45.32% | 4.68 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Daily | rf | RandomForest | 203 | 77 | 126 | 37.93% | 37.93% | 37.93% | 12.07 pp | -49 | 9 | -5.44 |
| BTC Daily | xgb | XGBoost | 213 | 76 | 137 | 35.68% | 35.68% | 35.68% | 14.32 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 203 | 68 | 135 | 33.50% | 33.50% | 33.50% | 16.50 pp | -67 | 9 | -7.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 201 | 102 | 99 | 50.75% | 50.75% | 50.75% | 0.75 pp | 3 | 16 | 0.19 |
| BTC Market Hours | transformer | Transformer | 201 | 98 | 103 | 48.76% | 48.76% | 48.76% | 1.24 pp | -5 | 16 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 201 | 94 | 107 | 46.77% | 46.77% | 46.77% | 3.23 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 201 | 91 | 110 | 45.27% | 45.27% | 45.27% | 4.73 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 201 | 86 | 115 | 42.79% | 42.79% | 42.79% | 7.21 pp | -29 | 16 | -1.81 |
| BTC Market Hours | lstm | LSTM | 201 | 84 | 117 | 41.79% | 41.79% | 41.79% | 8.21 pp | -33 | 16 | -2.06 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 201 | 104 | 97 | 51.74% | 51.74% | 51.74% | 1.74 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 17 | -0.41 |
| BTC Market Hours Daily | nn | NN | 201 | 96 | 105 | 47.76% | 47.76% | 47.76% | 2.24 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 201 | 88 | 113 | 43.78% | 43.78% | 43.78% | 6.22 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 201 | 82 | 119 | 40.80% | 40.80% | 40.80% | 9.20 pp | -37 | 17 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 201 | 81 | 120 | 40.30% | 40.30% | 40.30% | 9.70 pp | -39 | 17 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | rf | RandomForest | 173 | 84 | 89 | 48.55% | 48.55% | 48.55% | 1.45 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 173 | 78 | 95 | 45.09% | 45.09% | 45.09% | 4.91 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 173 | 85 | 88 | 49.13% | 49.13% | 49.13% | 0.87 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 173 | 84 | 89 | 48.55% | 48.55% | 48.55% | 1.45 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 173 | 79 | 94 | 45.66% | 45.66% | 45.66% | 4.34 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 173 | 78 | 95 | 45.09% | 45.09% | 45.09% | 4.91 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 173 | 77 | 96 | 44.51% | 44.51% | 44.51% | 5.49 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
