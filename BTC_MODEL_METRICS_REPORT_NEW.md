# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T06:44:52.573645+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 190 | 130 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 226 | 166 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 297 | 154 | 143 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 00:00:00+00:00 | 297 | 154 | 143 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 130 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-27T23:00:00+00:00 | 131 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 154 | 80 | 74 | 51.95% | 51.95% | 51.95% | 1.95 pp | 6 | 12 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 154 | 76 | 78 | 49.35% | 49.35% | 49.35% | 0.65 pp | -2 | 13 | -0.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 156 | 76 | 80 | 48.72% | 48.72% | 48.72% | 1.28 pp | -4 | 7 | -0.57 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 154 | 72 | 82 | 46.75% | 46.75% | 46.75% | 3.25 pp | -10 | 13 | -0.77 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours Daily | nn | NN | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 13 | -1.08 |
| BTC Market Hours | rf | RandomForest | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 12 | -1.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 154 | 69 | 85 | 44.81% | 44.81% | 44.81% | 5.19 pp | -16 | 12 | -1.33 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 154 | 66 | 88 | 42.86% | 42.86% | 42.86% | 7.14 pp | -22 | 13 | -1.69 |
| Consolidated Market Hours | lstm | LSTM | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 13 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 12 | -2.17 |
| BTC Hourly | nn | NN | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Market Hours | lstm | LSTM | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 12 | -2.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| BTC Daily | nn | NN | 156 | 69 | 87 | 44.23% | 44.23% | 44.23% | 5.77 pp | -18 | 7 | -2.57 |
| BTC Daily | transformer | Transformer | 156 | 68 | 88 | 43.59% | 43.59% | 43.59% | 6.41 pp | -20 | 7 | -2.86 |
| BTC Market Hours Daily | lstm | LSTM | 154 | 57 | 97 | 37.01% | 37.01% | 37.01% | 12.99 pp | -40 | 13 | -3.08 |
| BTC Hourly | rf | RandomForest | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 6 | -3.33 |
| BTC Daily | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours | nn | NN | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 2 | -5.00 |
| BTC Hourly | xgb | XGBoost | 130 | 49 | 81 | 37.69% | 37.69% | 37.69% | 12.31 pp | -32 | 6 | -5.33 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| BTC Daily | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 8 | -6.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 2 | -6.00 |
| BTC Hourly | lstm | LSTM | 130 | 46 | 84 | 35.38% | 35.38% | 35.38% | 14.62 pp | -38 | 6 | -6.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |
| BTC Daily | lstm | LSTM | 156 | 55 | 101 | 35.26% | 35.26% | 35.26% | 14.74 pp | -46 | 7 | -6.57 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | transformer | Transformer | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | nn | NN | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Hourly | rf | RandomForest | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 130 | 49 | 81 | 37.69% | 37.69% | 37.69% | 12.31 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 130 | 46 | 84 | 35.38% | 35.38% | 35.38% | 14.62 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 156 | 76 | 80 | 48.72% | 48.72% | 48.72% | 1.28 pp | -4 | 7 | -0.57 |
| BTC Daily | nn | NN | 156 | 69 | 87 | 44.23% | 44.23% | 44.23% | 5.77 pp | -18 | 7 | -2.57 |
| BTC Daily | transformer | Transformer | 156 | 68 | 88 | 43.59% | 43.59% | 43.59% | 6.41 pp | -20 | 7 | -2.86 |
| BTC Daily | rf | RandomForest | 156 | 65 | 91 | 41.67% | 41.67% | 41.67% | 8.33 pp | -26 | 7 | -3.71 |
| BTC Daily | xgb | XGBoost | 166 | 59 | 107 | 35.54% | 35.54% | 35.54% | 14.46 pp | -48 | 8 | -6.00 |
| BTC Daily | lstm | LSTM | 156 | 55 | 101 | 35.26% | 35.26% | 35.26% | 14.74 pp | -46 | 7 | -6.57 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 154 | 80 | 74 | 51.95% | 51.95% | 51.95% | 1.95 pp | 6 | 12 | 0.50 |
| BTC Market Hours | rf | RandomForest | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 12 | -1.17 |
| BTC Market Hours | transformer | Transformer | 154 | 70 | 84 | 45.45% | 45.45% | 45.45% | 4.55 pp | -14 | 12 | -1.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 154 | 69 | 85 | 44.81% | 44.81% | 44.81% | 5.19 pp | -16 | 12 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 154 | 64 | 90 | 41.56% | 41.56% | 41.56% | 8.44 pp | -26 | 12 | -2.17 |
| BTC Market Hours | lstm | LSTM | 154 | 63 | 91 | 40.91% | 40.91% | 40.91% | 9.09 pp | -28 | 12 | -2.33 |

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
| Consolidated Hourly | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Hourly | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 130 | 66 | 64 | 50.77% | 50.77% | 50.77% | 0.77 pp | 2 | 10 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 130 | 63 | 67 | 48.46% | 48.46% | 48.46% | 1.54 pp | -4 | 10 | -0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 130 | 60 | 70 | 46.15% | 46.15% | 46.15% | 3.85 pp | -10 | 10 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 130 | 57 | 73 | 43.85% | 43.85% | 43.85% | 6.15 pp | -16 | 10 | -1.60 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 27 | 13 | 14 | 48.15% | 48.15% | 48.15% | 1.85 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 27 | 12 | 15 | 44.44% | 44.44% | 44.44% | 5.56 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 27 | 8 | 19 | 29.63% | 29.63% | 29.63% | 20.37 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 2 | -6.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
