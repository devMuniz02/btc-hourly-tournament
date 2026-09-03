# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T07:37:13.255108+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 191 | 131 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 227 | 167 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 298 | 155 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 297 | 154 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 130 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 23:00:00+00:00 | 130 | 26 | 104 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | transformer | Transformer | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 13 | -0.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 7 | -0.43 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 13 | -0.77 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours Daily | rf | RandomForest | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 13 | -2.00 |
| BTC Hourly | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 6 | -2.17 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |
| BTC Daily | nn | NN | 157 | 70 | 87 | 44.59% | 44.59% | 44.59% | 5.41 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | lstm | LSTM | 154 | 57 | 97 | 37.01% | 37.01% | 37.01% | 12.99 pp | -40 | 13 | -3.08 |
| BTC Hourly | rf | RandomForest | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 6 | -3.17 |
| BTC Daily | rf | RandomForest | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 131 | 49 | 82 | 37.40% | 37.40% | 37.40% | 12.60 pp | -33 | 6 | -5.50 |
| BTC Daily | xgb | XGBoost | 167 | 60 | 107 | 35.93% | 35.93% | 35.93% | 14.07 pp | -47 | 8 | -5.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| BTC Hourly | lstm | LSTM | 131 | 47 | 84 | 35.88% | 35.88% | 35.88% | 14.12 pp | -37 | 6 | -6.17 |
| BTC Daily | lstm | LSTM | 157 | 55 | 102 | 35.03% | 35.03% | 35.03% | 14.97 pp | -47 | 7 | -6.71 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | transformer | Transformer | 131 | 66 | 65 | 50.38% | 50.38% | 50.38% | 0.38 pp | 1 | 6 | 0.17 |
| BTC Hourly | nn | NN | 131 | 59 | 72 | 45.04% | 45.04% | 45.04% | 4.96 pp | -13 | 6 | -2.17 |
| BTC Hourly | rf | RandomForest | 131 | 56 | 75 | 42.75% | 42.75% | 42.75% | 7.25 pp | -19 | 6 | -3.17 |
| BTC Hourly | xgb | XGBoost | 131 | 49 | 82 | 37.40% | 37.40% | 37.40% | 12.60 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 131 | 47 | 84 | 35.88% | 35.88% | 35.88% | 14.12 pp | -37 | 6 | -6.17 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 157 | 77 | 80 | 49.04% | 49.04% | 49.04% | 0.96 pp | -3 | 7 | -0.43 |
| BTC Daily | nn | NN | 157 | 70 | 87 | 44.59% | 44.59% | 44.59% | 5.41 pp | -17 | 7 | -2.43 |
| BTC Daily | transformer | Transformer | 157 | 69 | 88 | 43.95% | 43.95% | 43.95% | 6.05 pp | -19 | 7 | -2.71 |
| BTC Daily | rf | RandomForest | 157 | 66 | 91 | 42.04% | 42.04% | 42.04% | 7.96 pp | -25 | 7 | -3.57 |
| BTC Daily | xgb | XGBoost | 167 | 60 | 107 | 35.93% | 35.93% | 35.93% | 14.07 pp | -47 | 8 | -5.88 |
| BTC Daily | lstm | LSTM | 157 | 55 | 102 | 35.03% | 35.03% | 35.03% | 14.97 pp | -47 | 7 | -6.71 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 155 | 81 | 74 | 52.26% | 52.26% | 52.26% | 2.26 pp | 7 | 12 | 0.58 |
| BTC Market Hours | transformer | Transformer | 155 | 71 | 84 | 45.81% | 45.81% | 45.81% | 4.19 pp | -13 | 12 | -1.08 |
| BTC Market Hours | rf | RandomForest | 155 | 70 | 85 | 45.16% | 45.16% | 45.16% | 4.84 pp | -15 | 12 | -1.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| BTC Market Hours | lstm | LSTM | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 13 | -0.15 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 13 | -0.77 |
| BTC Market Hours Daily | nn | NN | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 13 | -1.69 |
| BTC Market Hours Daily | xgb | XGBoost | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 13 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 154 | 57 | 97 | 37.01% | 37.01% | 37.01% | 12.99 pp | -40 | 13 | -3.08 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 68 | 62 | 52.31% | 52.31% | 52.31% | 2.31 pp | 6 | 10 | 0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 10 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 53 | 77 | 40.77% | 40.77% | 40.77% | 9.23 pp | -24 | 10 | -2.40 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
