# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T00:33:00.140295+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 218 | 158 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 254 | 194 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-04 23:00:00+00:00 | 350 | 182 | 168 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-04 23:00:00+00:00 | 350 | 182 | 168 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 155 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T00:00:00+00:00 | 156 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 158 | 82 | 76 | 51.90% | 51.90% | 51.90% | 1.90 pp | 6 | 7 | 0.86 |
| BTC Market Hours Daily | transformer | Transformer | 182 | 95 | 87 | 52.20% | 52.20% | 52.20% | 2.20 pp | 8 | 15 | 0.53 |
| BTC Market Hours | nn | NN | 182 | 93 | 89 | 51.10% | 51.10% | 51.10% | 1.10 pp | 4 | 14 | 0.29 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| BTC Market Hours | transformer | Transformer | 182 | 89 | 93 | 48.90% | 48.90% | 48.90% | 1.10 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 182 | 88 | 94 | 48.35% | 48.35% | 48.35% | 1.65 pp | -6 | 15 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 15 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| BTC Hourly | transformer | Transformer | 158 | 76 | 82 | 48.10% | 48.10% | 48.10% | 1.90 pp | -6 | 7 | -0.86 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| BTC Market Hours | rf | RandomForest | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 15 | -1.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 182 | 75 | 107 | 41.21% | 41.21% | 41.21% | 8.79 pp | -32 | 15 | -2.13 |
| BTC Daily | nn | NN | 184 | 83 | 101 | 45.11% | 45.11% | 45.11% | 4.89 pp | -18 | 8 | -2.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 15 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 184 | 80 | 104 | 43.48% | 43.48% | 43.48% | 6.52 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 158 | 67 | 91 | 42.41% | 42.41% | 42.41% | 7.59 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 7 | -4.00 |
| BTC Daily | rf | RandomForest | 184 | 73 | 111 | 39.67% | 39.67% | 39.67% | 10.33 pp | -38 | 8 | -4.75 |
| BTC Daily | xgb | XGBoost | 194 | 70 | 124 | 36.08% | 36.08% | 36.08% | 13.92 pp | -54 | 9 | -6.00 |
| BTC Hourly | lstm | LSTM | 158 | 57 | 101 | 36.08% | 36.08% | 36.08% | 13.92 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 158 | 56 | 102 | 35.44% | 35.44% | 35.44% | 14.56 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 184 | 63 | 121 | 34.24% | 34.24% | 34.24% | 15.76 pp | -58 | 8 | -7.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 158 | 82 | 76 | 51.90% | 51.90% | 51.90% | 1.90 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 158 | 76 | 82 | 48.10% | 48.10% | 48.10% | 1.90 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 158 | 67 | 91 | 42.41% | 42.41% | 42.41% | 7.59 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 158 | 65 | 93 | 41.14% | 41.14% | 41.14% | 8.86 pp | -28 | 7 | -4.00 |
| BTC Hourly | lstm | LSTM | 158 | 57 | 101 | 36.08% | 36.08% | 36.08% | 13.92 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 158 | 56 | 102 | 35.44% | 35.44% | 35.44% | 14.56 pp | -46 | 7 | -6.57 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 8 | -1.50 |
| BTC Daily | nn | NN | 184 | 83 | 101 | 45.11% | 45.11% | 45.11% | 4.89 pp | -18 | 8 | -2.25 |
| BTC Daily | transformer | Transformer | 184 | 80 | 104 | 43.48% | 43.48% | 43.48% | 6.52 pp | -24 | 8 | -3.00 |
| BTC Daily | rf | RandomForest | 184 | 73 | 111 | 39.67% | 39.67% | 39.67% | 10.33 pp | -38 | 8 | -4.75 |
| BTC Daily | xgb | XGBoost | 194 | 70 | 124 | 36.08% | 36.08% | 36.08% | 13.92 pp | -54 | 9 | -6.00 |
| BTC Daily | lstm | LSTM | 184 | 63 | 121 | 34.24% | 34.24% | 34.24% | 15.76 pp | -58 | 8 | -7.25 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 182 | 93 | 89 | 51.10% | 51.10% | 51.10% | 1.10 pp | 4 | 14 | 0.29 |
| BTC Market Hours | transformer | Transformer | 182 | 89 | 93 | 48.90% | 48.90% | 48.90% | 1.10 pp | -4 | 14 | -0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| BTC Market Hours | rf | RandomForest | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| BTC Market Hours | lstm | LSTM | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 182 | 95 | 87 | 52.20% | 52.20% | 52.20% | 2.20 pp | 8 | 15 | 0.53 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 182 | 88 | 94 | 48.35% | 48.35% | 48.35% | 1.65 pp | -6 | 15 | -0.40 |
| BTC Market Hours Daily | nn | NN | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 15 | -0.67 |
| BTC Market Hours Daily | rf | RandomForest | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 15 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 182 | 75 | 107 | 41.21% | 41.21% | 41.21% | 8.79 pp | -32 | 15 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 15 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Hourly | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 77 | 78 | 49.68% | 49.68% | 49.68% | 0.32 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 73 | 82 | 47.10% | 47.10% | 47.10% | 2.90 pp | -9 | 12 | -0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 66 | 89 | 42.58% | 42.58% | 42.58% | 7.42 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 41 | 20 | 21 | 48.78% | 48.78% | 48.78% | 1.22 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 41 | 18 | 23 | 43.90% | 43.90% | 43.90% | 6.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 41 | 16 | 25 | 39.02% | 39.02% | 39.02% | 10.98 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 41 | 15 | 26 | 36.59% | 36.59% | 36.59% | 13.41 pp | -11 | 4 | -2.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
