# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T15:23:17.724378+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 261 | 201 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 297 | 237 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 14:00:00+00:00 | 423 | 225 | 198 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 14:00:00+00:00 | 423 | 225 | 198 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 195 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 195 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 195 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T19:00:00+00:00 | 196 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 225 | 117 | 108 | 52.00% | 52.00% | 52.00% | 2.00 pp | 9 | 18 | 0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | transformer | Transformer | 225 | 112 | 113 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 19 | -0.47 |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| BTC Market Hours | transformer | Transformer | 225 | 107 | 118 | 47.56% | 47.56% | 47.56% | 2.44 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | nn | NN | 225 | 106 | 119 | 47.11% | 47.11% | 47.11% | 2.89 pp | -13 | 19 | -0.68 |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 225 | 105 | 120 | 46.67% | 46.67% | 46.67% | 3.33 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 225 | 103 | 122 | 45.78% | 45.78% | 45.78% | 4.22 pp | -19 | 18 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 19 | -1.32 |
| BTC Market Hours | xgb | XGBoost | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 18 | -1.39 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | lstm | LSTM | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| BTC Daily | mlp_sklearn | MLPClassifier | 227 | 104 | 123 | 45.81% | 45.81% | 45.81% | 4.19 pp | -19 | 10 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |
| BTC Market Hours Daily | xgb | XGBoost | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 19 | -1.95 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 225 | 93 | 132 | 41.33% | 41.33% | 41.33% | 8.67 pp | -39 | 18 | -2.17 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| BTC Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 225 | 89 | 136 | 39.56% | 39.56% | 39.56% | 10.44 pp | -47 | 19 | -2.47 |
| BTC Daily | nn | NN | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 201 | 85 | 116 | 42.29% | 42.29% | 42.29% | 7.71 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 201 | 84 | 117 | 41.79% | 41.79% | 41.79% | 8.21 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 10 | -4.50 |
| BTC Daily | rf | RandomForest | 227 | 87 | 140 | 38.33% | 38.33% | 38.33% | 11.67 pp | -53 | 10 | -5.30 |
| BTC Hourly | lstm | LSTM | 201 | 76 | 125 | 37.81% | 37.81% | 37.81% | 12.19 pp | -49 | 9 | -5.44 |
| BTC Daily | xgb | XGBoost | 237 | 83 | 154 | 35.02% | 35.02% | 35.02% | 14.98 pp | -71 | 11 | -6.45 |
| BTC Hourly | xgb | XGBoost | 201 | 71 | 130 | 35.32% | 35.32% | 35.32% | 14.68 pp | -59 | 9 | -6.56 |
| BTC Daily | lstm | LSTM | 227 | 75 | 152 | 33.04% | 33.04% | 33.04% | 16.96 pp | -77 | 10 | -7.70 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 201 | 101 | 100 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 9 | 0.11 |
| BTC Hourly | transformer | Transformer | 201 | 90 | 111 | 44.78% | 44.78% | 44.78% | 5.22 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 201 | 85 | 116 | 42.29% | 42.29% | 42.29% | 7.71 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 201 | 84 | 117 | 41.79% | 41.79% | 41.79% | 8.21 pp | -33 | 9 | -3.67 |
| BTC Hourly | lstm | LSTM | 201 | 76 | 125 | 37.81% | 37.81% | 37.81% | 12.19 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 201 | 71 | 130 | 35.32% | 35.32% | 35.32% | 14.68 pp | -59 | 9 | -6.56 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 227 | 104 | 123 | 45.81% | 45.81% | 45.81% | 4.19 pp | -19 | 10 | -1.90 |
| BTC Daily | nn | NN | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 10 | -2.50 |
| BTC Daily | transformer | Transformer | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 10 | -4.50 |
| BTC Daily | rf | RandomForest | 227 | 87 | 140 | 38.33% | 38.33% | 38.33% | 11.67 pp | -53 | 10 | -5.30 |
| BTC Daily | xgb | XGBoost | 237 | 83 | 154 | 35.02% | 35.02% | 35.02% | 14.98 pp | -71 | 11 | -6.45 |
| BTC Daily | lstm | LSTM | 227 | 75 | 152 | 33.04% | 33.04% | 33.04% | 16.96 pp | -77 | 10 | -7.70 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 225 | 117 | 108 | 52.00% | 52.00% | 52.00% | 2.00 pp | 9 | 18 | 0.50 |
| BTC Market Hours | transformer | Transformer | 225 | 107 | 118 | 47.56% | 47.56% | 47.56% | 2.44 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 225 | 105 | 120 | 46.67% | 46.67% | 46.67% | 3.33 pp | -15 | 18 | -0.83 |
| BTC Market Hours | rf | RandomForest | 225 | 103 | 122 | 45.78% | 45.78% | 45.78% | 4.22 pp | -19 | 18 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 18 | -1.39 |
| BTC Market Hours | lstm | LSTM | 225 | 93 | 132 | 41.33% | 41.33% | 41.33% | 8.67 pp | -39 | 18 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 225 | 112 | 113 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 225 | 108 | 117 | 48.00% | 48.00% | 48.00% | 2.00 pp | -9 | 19 | -0.47 |
| BTC Market Hours Daily | nn | NN | 225 | 106 | 119 | 47.11% | 47.11% | 47.11% | 2.89 pp | -13 | 19 | -0.68 |
| BTC Market Hours Daily | rf | RandomForest | 225 | 100 | 125 | 44.44% | 44.44% | 44.44% | 5.56 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 225 | 94 | 131 | 41.78% | 41.78% | 41.78% | 8.22 pp | -37 | 19 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 225 | 89 | 136 | 39.56% | 39.56% | 39.56% | 10.44 pp | -47 | 19 | -2.47 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | rf | RandomForest | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | xgb | XGBoost | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | lstm | LSTM | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 195 | 96 | 99 | 49.23% | 49.23% | 49.23% | 0.77 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 195 | 89 | 106 | 45.64% | 45.64% | 45.64% | 4.36 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 13 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 61 | 29 | 32 | 47.54% | 47.54% | 47.54% | 2.46 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | rf | RandomForest | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 61 | 27 | 34 | 44.26% | 44.26% | 44.26% | 5.74 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | nn | NN | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 61 | 24 | 37 | 39.34% | 39.34% | 39.34% | 10.66 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 29 | 33 | 46.77% | 46.77% | 46.77% | 3.23 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
