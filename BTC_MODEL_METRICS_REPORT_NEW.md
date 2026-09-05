# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T01:10:32.028978+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 351 | 182 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 351 | 182 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 158 | 83 | 75 | 52.53% | 52.53% | 52.53% | 2.53 pp | 8 | 7 | 1.14 |
| BTC Market Hours Daily | transformer | Transformer | 182 | 95 | 87 | 52.20% | 52.20% | 52.20% | 2.20 pp | 8 | 15 | 0.53 |
| BTC Market Hours | nn | NN | 182 | 93 | 89 | 51.10% | 51.10% | 51.10% | 1.10 pp | 4 | 14 | 0.29 |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| BTC Market Hours | transformer | Transformer | 182 | 89 | 93 | 48.90% | 48.90% | 48.90% | 1.10 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 182 | 88 | 94 | 48.35% | 48.35% | 48.35% | 1.65 pp | -6 | 15 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Hourly | transformer | Transformer | 158 | 77 | 81 | 48.73% | 48.73% | 48.73% | 1.27 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | nn | NN | 182 | 86 | 96 | 47.25% | 47.25% | 47.25% | 2.75 pp | -10 | 15 | -0.67 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| BTC Market Hours | rf | RandomForest | 182 | 84 | 98 | 46.15% | 46.15% | 46.15% | 3.85 pp | -14 | 14 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 182 | 80 | 102 | 43.96% | 43.96% | 43.96% | 6.04 pp | -22 | 15 | -1.47 |
| BTC Daily | mlp_sklearn | MLPClassifier | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| BTC Market Hours | lstm | LSTM | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 182 | 77 | 105 | 42.31% | 42.31% | 42.31% | 7.69 pp | -28 | 14 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 182 | 75 | 107 | 41.21% | 41.21% | 41.21% | 8.79 pp | -32 | 15 | -2.13 |
| BTC Daily | nn | NN | 184 | 83 | 101 | 45.11% | 45.11% | 45.11% | 4.89 pp | -18 | 8 | -2.25 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 182 | 73 | 109 | 40.11% | 40.11% | 40.11% | 9.89 pp | -36 | 15 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 184 | 80 | 104 | 43.48% | 43.48% | 43.48% | 6.52 pp | -24 | 8 | -3.00 |
| BTC Hourly | nn | NN | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 158 | 66 | 92 | 41.77% | 41.77% | 41.77% | 8.23 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 184 | 73 | 111 | 39.67% | 39.67% | 39.67% | 10.33 pp | -38 | 8 | -4.75 |
| BTC Hourly | lstm | LSTM | 158 | 58 | 100 | 36.71% | 36.71% | 36.71% | 13.29 pp | -42 | 7 | -6.00 |
| BTC Daily | xgb | XGBoost | 194 | 70 | 124 | 36.08% | 36.08% | 36.08% | 13.92 pp | -54 | 9 | -6.00 |
| BTC Hourly | xgb | XGBoost | 158 | 56 | 102 | 35.44% | 35.44% | 35.44% | 14.56 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 184 | 63 | 121 | 34.24% | 34.24% | 34.24% | 15.76 pp | -58 | 8 | -7.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 158 | 83 | 75 | 52.53% | 52.53% | 52.53% | 2.53 pp | 8 | 7 | 1.14 |
| BTC Hourly | transformer | Transformer | 158 | 77 | 81 | 48.73% | 48.73% | 48.73% | 1.27 pp | -4 | 7 | -0.57 |
| BTC Hourly | nn | NN | 158 | 68 | 90 | 43.04% | 43.04% | 43.04% | 6.96 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 158 | 66 | 92 | 41.77% | 41.77% | 41.77% | 8.23 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 158 | 58 | 100 | 36.71% | 36.71% | 36.71% | 13.29 pp | -42 | 7 | -6.00 |
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
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
