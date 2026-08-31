# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T22:39:27.334317+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 152 | 92 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 188 | 128 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 21:00:00+00:00 | 230 | 116 | 114 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 21:00:00+00:00 | 230 | 116 | 114 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 95 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T17:00:00+00:00 | 96 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 116 | 63 | 53 | 54.31% | 54.31% | 54.31% | 4.31 pp | 10 | 9 | 1.11 |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| BTC Hourly | transformer | Transformer | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 9 | -0.67 |
| BTC Hourly | nn | NN | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | nn | NN | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | transformer | Transformer | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 10 | -1.20 |
| BTC Daily | nn | NN | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 116 | 51 | 65 | 43.97% | 43.97% | 43.97% | 6.03 pp | -14 | 10 | -1.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 4 | -1.50 |
| BTC Daily | transformer | Transformer | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 6 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Market Hours | transformer | Transformer | 116 | 48 | 68 | 41.38% | 41.38% | 41.38% | 8.62 pp | -20 | 9 | -2.22 |
| BTC Market Hours | xgb | XGBoost | 116 | 47 | 69 | 40.52% | 40.52% | 40.52% | 9.48 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 116 | 45 | 71 | 38.79% | 38.79% | 38.79% | 11.21 pp | -26 | 10 | -2.60 |
| BTC Market Hours | lstm | LSTM | 116 | 45 | 71 | 38.79% | 38.79% | 38.79% | 11.21 pp | -26 | 9 | -2.89 |
| BTC Market Hours Daily | lstm | LSTM | 116 | 43 | 73 | 37.07% | 37.07% | 37.07% | 12.93 pp | -30 | 10 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 118 | 47 | 71 | 39.83% | 39.83% | 39.83% | 10.17 pp | -24 | 6 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 4 | -4.50 |
| BTC Daily | xgb | XGBoost | 128 | 46 | 82 | 35.94% | 35.94% | 35.94% | 14.06 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 118 | 41 | 77 | 34.75% | 34.75% | 34.75% | 15.25 pp | -36 | 6 | -6.00 |
| BTC Hourly | lstm | LSTM | 92 | 30 | 62 | 32.61% | 32.61% | 32.61% | 17.39 pp | -32 | 4 | -8.00 |
| BTC Hourly | xgb | XGBoost | 92 | 30 | 62 | 32.61% | 32.61% | 32.61% | 17.39 pp | -32 | 4 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 4 | -0.50 |
| BTC Hourly | nn | NN | 92 | 44 | 48 | 47.83% | 47.83% | 47.83% | 2.17 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 4 | -4.50 |
| BTC Hourly | lstm | LSTM | 92 | 30 | 62 | 32.61% | 32.61% | 32.61% | 17.39 pp | -32 | 4 | -8.00 |
| BTC Hourly | xgb | XGBoost | 92 | 30 | 62 | 32.61% | 32.61% | 32.61% | 17.39 pp | -32 | 4 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 118 | 57 | 61 | 48.31% | 48.31% | 48.31% | 1.69 pp | -4 | 6 | -0.67 |
| BTC Daily | nn | NN | 118 | 55 | 63 | 46.61% | 46.61% | 46.61% | 3.39 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 118 | 53 | 65 | 44.92% | 44.92% | 44.92% | 5.08 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 118 | 47 | 71 | 39.83% | 39.83% | 39.83% | 10.17 pp | -24 | 6 | -4.00 |
| BTC Daily | xgb | XGBoost | 128 | 46 | 82 | 35.94% | 35.94% | 35.94% | 14.06 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 118 | 41 | 77 | 34.75% | 34.75% | 34.75% | 15.25 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 116 | 63 | 53 | 54.31% | 54.31% | 54.31% | 4.31 pp | 10 | 9 | 1.11 |
| BTC Market Hours | rf | RandomForest | 116 | 55 | 61 | 47.41% | 47.41% | 47.41% | 2.59 pp | -6 | 9 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 116 | 53 | 63 | 45.69% | 45.69% | 45.69% | 4.31 pp | -10 | 9 | -1.11 |
| BTC Market Hours | transformer | Transformer | 116 | 48 | 68 | 41.38% | 41.38% | 41.38% | 8.62 pp | -20 | 9 | -2.22 |
| BTC Market Hours | xgb | XGBoost | 116 | 47 | 69 | 40.52% | 40.52% | 40.52% | 9.48 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 116 | 45 | 71 | 38.79% | 38.79% | 38.79% | 11.21 pp | -26 | 9 | -2.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 116 | 57 | 59 | 49.14% | 49.14% | 49.14% | 0.86 pp | -2 | 10 | -0.20 |
| BTC Market Hours Daily | nn | NN | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | transformer | Transformer | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 116 | 51 | 65 | 43.97% | 43.97% | 43.97% | 6.03 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 116 | 45 | 71 | 38.79% | 38.79% | 38.79% | 11.21 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 116 | 43 | 73 | 37.07% | 37.07% | 37.07% | 12.93 pp | -30 | 10 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 95 | 50 | 45 | 52.63% | 52.63% | 52.63% | 2.63 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 95 | 49 | 46 | 51.58% | 51.58% | 51.58% | 1.58 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 7 | 5 | 2 | 71.43% | 71.43% | 71.43% | 21.43 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 7 | 3 | 4 | 42.86% | 42.86% | 42.86% | 7.14 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 8 | 7 | 1 | 87.50% | 87.50% | 87.50% | 37.50 pp | 6 | 1 | 6.00 |
| Consolidated Market Hours Daily | nn | NN | 8 | 6 | 2 | 75.00% | 75.00% | 75.00% | 25.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 8 | 5 | 3 | 62.50% | 62.50% | 62.50% | 12.50 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 8 | 2 | 6 | 25.00% | 25.00% | 25.00% | 25.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
