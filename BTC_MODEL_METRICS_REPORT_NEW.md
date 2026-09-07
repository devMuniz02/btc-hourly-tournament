# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T13:13:17.555074+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 260 | 200 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 295 | 235 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 419 | 223 | 196 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 12:00:00+00:00 | 419 | 223 | 196 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 193 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 193 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 60 | 133 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 18:00:00+00:00 | 193 | 60 | 133 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 223 | 116 | 107 | 52.02% | 52.02% | 52.02% | 2.02 pp | 9 | 18 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 223 | 112 | 111 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 19 | 0.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 100 | 100 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 223 | 107 | 116 | 47.98% | 47.98% | 47.98% | 2.02 pp | -9 | 19 | -0.47 |
| BTC Market Hours | transformer | Transformer | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | nn | NN | 223 | 104 | 119 | 46.64% | 46.64% | 46.64% | 3.36 pp | -15 | 19 | -0.79 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 223 | 104 | 119 | 46.64% | 46.64% | 46.64% | 3.36 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 18 | -0.94 |
| Consolidated Hourly | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 19 | -1.21 |
| BTC Market Hours | xgb | XGBoost | 223 | 99 | 124 | 44.39% | 44.39% | 44.39% | 5.61 pp | -25 | 18 | -1.39 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 225 | 103 | 122 | 45.78% | 45.78% | 45.78% | 4.22 pp | -19 | 10 | -1.90 |
| BTC Market Hours Daily | xgb | XGBoost | 223 | 93 | 130 | 41.70% | 41.70% | 41.70% | 8.30 pp | -37 | 19 | -1.95 |
| Consolidated Hourly | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 223 | 91 | 132 | 40.81% | 40.81% | 40.81% | 9.19 pp | -41 | 18 | -2.28 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| BTC Hourly | transformer | Transformer | 200 | 89 | 111 | 44.50% | 44.50% | 44.50% | 5.50 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | lstm | LSTM | 223 | 87 | 136 | 39.01% | 39.01% | 39.01% | 10.99 pp | -49 | 19 | -2.58 |
| BTC Daily | nn | NN | 225 | 99 | 126 | 44.00% | 44.00% | 44.00% | 6.00 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 83 | 117 | 41.50% | 41.50% | 41.50% | 8.50 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 225 | 90 | 135 | 40.00% | 40.00% | 40.00% | 10.00 pp | -45 | 10 | -4.50 |
| BTC Hourly | lstm | LSTM | 200 | 76 | 124 | 38.00% | 38.00% | 38.00% | 12.00 pp | -48 | 9 | -5.33 |
| BTC Daily | rf | RandomForest | 225 | 85 | 140 | 37.78% | 37.78% | 37.78% | 12.22 pp | -55 | 10 | -5.50 |
| BTC Hourly | xgb | XGBoost | 200 | 71 | 129 | 35.50% | 35.50% | 35.50% | 14.50 pp | -58 | 9 | -6.44 |
| BTC Daily | xgb | XGBoost | 235 | 82 | 153 | 34.89% | 34.89% | 34.89% | 15.11 pp | -71 | 11 | -6.45 |
| BTC Daily | lstm | LSTM | 225 | 75 | 150 | 33.33% | 33.33% | 33.33% | 16.67 pp | -75 | 10 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 200 | 100 | 100 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 200 | 89 | 111 | 44.50% | 44.50% | 44.50% | 5.50 pp | -22 | 9 | -2.44 |
| BTC Hourly | nn | NN | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | rf | RandomForest | 200 | 83 | 117 | 41.50% | 41.50% | 41.50% | 8.50 pp | -34 | 9 | -3.78 |
| BTC Hourly | lstm | LSTM | 200 | 76 | 124 | 38.00% | 38.00% | 38.00% | 12.00 pp | -48 | 9 | -5.33 |
| BTC Hourly | xgb | XGBoost | 200 | 71 | 129 | 35.50% | 35.50% | 35.50% | 14.50 pp | -58 | 9 | -6.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 225 | 103 | 122 | 45.78% | 45.78% | 45.78% | 4.22 pp | -19 | 10 | -1.90 |
| BTC Daily | nn | NN | 225 | 99 | 126 | 44.00% | 44.00% | 44.00% | 6.00 pp | -27 | 10 | -2.70 |
| BTC Daily | transformer | Transformer | 225 | 90 | 135 | 40.00% | 40.00% | 40.00% | 10.00 pp | -45 | 10 | -4.50 |
| BTC Daily | rf | RandomForest | 225 | 85 | 140 | 37.78% | 37.78% | 37.78% | 12.22 pp | -55 | 10 | -5.50 |
| BTC Daily | xgb | XGBoost | 235 | 82 | 153 | 34.89% | 34.89% | 34.89% | 15.11 pp | -71 | 11 | -6.45 |
| BTC Daily | lstm | LSTM | 225 | 75 | 150 | 33.33% | 33.33% | 33.33% | 16.67 pp | -75 | 10 | -7.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 223 | 116 | 107 | 52.02% | 52.02% | 52.02% | 2.02 pp | 9 | 18 | 0.50 |
| BTC Market Hours | transformer | Transformer | 223 | 106 | 117 | 47.53% | 47.53% | 47.53% | 2.47 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 223 | 104 | 119 | 46.64% | 46.64% | 46.64% | 3.36 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 18 | -0.94 |
| BTC Market Hours | xgb | XGBoost | 223 | 99 | 124 | 44.39% | 44.39% | 44.39% | 5.61 pp | -25 | 18 | -1.39 |
| BTC Market Hours | lstm | LSTM | 223 | 91 | 132 | 40.81% | 40.81% | 40.81% | 9.19 pp | -41 | 18 | -2.28 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 223 | 112 | 111 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 19 | 0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 223 | 107 | 116 | 47.98% | 47.98% | 47.98% | 2.02 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | nn | NN | 223 | 104 | 119 | 46.64% | 46.64% | 46.64% | 3.36 pp | -15 | 19 | -0.79 |
| BTC Market Hours Daily | rf | RandomForest | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 19 | -1.21 |
| BTC Market Hours Daily | xgb | XGBoost | 223 | 93 | 130 | 41.70% | 41.70% | 41.70% | 8.30 pp | -37 | 19 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 223 | 87 | 136 | 39.01% | 39.01% | 39.01% | 10.99 pp | -49 | 19 | -2.58 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 193 | 95 | 98 | 49.22% | 49.22% | 49.22% | 0.78 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 193 | 89 | 104 | 46.11% | 46.11% | 46.11% | 3.89 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 193 | 85 | 108 | 44.04% | 44.04% | 44.04% | 5.96 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 60 | 23 | 37 | 38.33% | 38.33% | 38.33% | 11.67 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
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
