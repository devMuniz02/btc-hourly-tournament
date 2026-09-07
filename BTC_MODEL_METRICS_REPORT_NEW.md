# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T05:44:24.238938+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 255 | 195 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 290 | 230 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 413 | 218 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 413 | 218 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 16:00:00+00:00 | 189 | 189 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 16:00:00+00:00 | 189 | 189 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 16:00:00+00:00 | 189 | 58 | 131 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 16:00:00+00:00 | 189 | 58 | 131 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 218 | 114 | 104 | 52.29% | 52.29% | 52.29% | 2.29 pp | 10 | 17 | 0.59 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 195 | 99 | 96 | 50.77% | 50.77% | 50.77% | 0.77 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 189 | 95 | 94 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 189 | 95 | 94 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Market Hours | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| BTC Market Hours Daily | nn | NN | 218 | 107 | 111 | 49.08% | 49.08% | 49.08% | 0.92 pp | -4 | 18 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 218 | 105 | 113 | 48.17% | 48.17% | 48.17% | 1.83 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 18 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 218 | 103 | 115 | 47.25% | 47.25% | 47.25% | 2.75 pp | -12 | 17 | -0.71 |
| Consolidated Hourly | lstm | LSTM | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| BTC Market Hours | transformer | Transformer | 218 | 100 | 118 | 45.87% | 45.87% | 45.87% | 4.13 pp | -18 | 17 | -1.06 |
| BTC Market Hours Daily | rf | RandomForest | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 18 | -1.11 |
| Consolidated Market Hours | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| BTC Daily | mlp_sklearn | MLPClassifier | 220 | 102 | 118 | 46.36% | 46.36% | 46.36% | 3.64 pp | -16 | 10 | -1.60 |
| Consolidated Market Hours | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 218 | 93 | 125 | 42.66% | 42.66% | 42.66% | 7.34 pp | -32 | 17 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 218 | 92 | 126 | 42.20% | 42.20% | 42.20% | 7.80 pp | -34 | 18 | -1.89 |
| Consolidated Hourly | nn | NN | 189 | 81 | 108 | 42.86% | 42.86% | 42.86% | 7.14 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 189 | 81 | 108 | 42.86% | 42.86% | 42.86% | 7.14 pp | -27 | 13 | -2.08 |
| BTC Hourly | transformer | Transformer | 195 | 88 | 107 | 45.13% | 45.13% | 45.13% | 4.87 pp | -19 | 9 | -2.11 |
| BTC Daily | nn | NN | 220 | 98 | 122 | 44.55% | 44.55% | 44.55% | 5.45 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 195 | 83 | 112 | 42.56% | 42.56% | 42.56% | 7.44 pp | -29 | 9 | -3.22 |
| BTC Hourly | rf | RandomForest | 195 | 83 | 112 | 42.56% | 42.56% | 42.56% | 7.44 pp | -29 | 9 | -3.22 |
| BTC Market Hours Daily | lstm | LSTM | 218 | 76 | 142 | 34.86% | 34.86% | 34.86% | 15.14 pp | -66 | 18 | -3.67 |
| BTC Market Hours | lstm | LSTM | 218 | 77 | 141 | 35.32% | 35.32% | 35.32% | 14.68 pp | -64 | 17 | -3.76 |
| BTC Daily | transformer | Transformer | 220 | 89 | 131 | 40.45% | 40.45% | 40.45% | 9.55 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 220 | 84 | 136 | 38.18% | 38.18% | 38.18% | 11.82 pp | -52 | 10 | -5.20 |
| BTC Hourly | lstm | LSTM | 195 | 73 | 122 | 37.44% | 37.44% | 37.44% | 12.56 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 195 | 71 | 124 | 36.41% | 36.41% | 36.41% | 13.59 pp | -53 | 9 | -5.89 |
| BTC Daily | xgb | XGBoost | 230 | 81 | 149 | 35.22% | 35.22% | 35.22% | 14.78 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 220 | 74 | 146 | 33.64% | 33.64% | 33.64% | 16.36 pp | -72 | 10 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 195 | 99 | 96 | 50.77% | 50.77% | 50.77% | 0.77 pp | 3 | 9 | 0.33 |
| BTC Hourly | transformer | Transformer | 195 | 88 | 107 | 45.13% | 45.13% | 45.13% | 4.87 pp | -19 | 9 | -2.11 |
| BTC Hourly | nn | NN | 195 | 83 | 112 | 42.56% | 42.56% | 42.56% | 7.44 pp | -29 | 9 | -3.22 |
| BTC Hourly | rf | RandomForest | 195 | 83 | 112 | 42.56% | 42.56% | 42.56% | 7.44 pp | -29 | 9 | -3.22 |
| BTC Hourly | lstm | LSTM | 195 | 73 | 122 | 37.44% | 37.44% | 37.44% | 12.56 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 195 | 71 | 124 | 36.41% | 36.41% | 36.41% | 13.59 pp | -53 | 9 | -5.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 220 | 102 | 118 | 46.36% | 46.36% | 46.36% | 3.64 pp | -16 | 10 | -1.60 |
| BTC Daily | nn | NN | 220 | 98 | 122 | 44.55% | 44.55% | 44.55% | 5.45 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 220 | 89 | 131 | 40.45% | 40.45% | 40.45% | 9.55 pp | -42 | 10 | -4.20 |
| BTC Daily | rf | RandomForest | 220 | 84 | 136 | 38.18% | 38.18% | 38.18% | 11.82 pp | -52 | 10 | -5.20 |
| BTC Daily | xgb | XGBoost | 230 | 81 | 149 | 35.22% | 35.22% | 35.22% | 14.78 pp | -68 | 11 | -6.18 |
| BTC Daily | lstm | LSTM | 220 | 74 | 146 | 33.64% | 33.64% | 33.64% | 16.36 pp | -72 | 10 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 218 | 114 | 104 | 52.29% | 52.29% | 52.29% | 2.29 pp | 10 | 17 | 0.59 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 17 | -0.59 |
| BTC Market Hours | rf | RandomForest | 218 | 103 | 115 | 47.25% | 47.25% | 47.25% | 2.75 pp | -12 | 17 | -0.71 |
| BTC Market Hours | transformer | Transformer | 218 | 100 | 118 | 45.87% | 45.87% | 45.87% | 4.13 pp | -18 | 17 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 218 | 93 | 125 | 42.66% | 42.66% | 42.66% | 7.34 pp | -32 | 17 | -1.88 |
| BTC Market Hours | lstm | LSTM | 218 | 77 | 141 | 35.32% | 35.32% | 35.32% | 14.68 pp | -64 | 17 | -3.76 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 218 | 107 | 111 | 49.08% | 49.08% | 49.08% | 0.92 pp | -4 | 18 | -0.22 |
| BTC Market Hours Daily | transformer | Transformer | 218 | 105 | 113 | 48.17% | 48.17% | 48.17% | 1.83 pp | -8 | 18 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 218 | 104 | 114 | 47.71% | 47.71% | 47.71% | 2.29 pp | -10 | 18 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 218 | 99 | 119 | 45.41% | 45.41% | 45.41% | 4.59 pp | -20 | 18 | -1.11 |
| BTC Market Hours Daily | xgb | XGBoost | 218 | 92 | 126 | 42.20% | 42.20% | 42.20% | 7.80 pp | -34 | 18 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 218 | 76 | 142 | 34.86% | 34.86% | 34.86% | 15.14 pp | -66 | 18 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 189 | 95 | 94 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | lstm | LSTM | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | xgb | XGBoost | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | nn | NN | 189 | 81 | 108 | 42.86% | 42.86% | 42.86% | 7.14 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 189 | 95 | 94 | 50.26% | 50.26% | 50.26% | 0.26 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 189 | 94 | 95 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 189 | 88 | 101 | 46.56% | 46.56% | 46.56% | 3.44 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 189 | 84 | 105 | 44.44% | 44.44% | 44.44% | 5.56 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 189 | 81 | 108 | 42.86% | 42.86% | 42.86% | 7.14 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 58 | 29 | 29 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 58 | 26 | 32 | 44.83% | 44.83% | 44.83% | 5.17 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 58 | 25 | 33 | 43.10% | 43.10% | 43.10% | 6.90 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 58 | 23 | 35 | 39.66% | 39.66% | 39.66% | 10.34 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
