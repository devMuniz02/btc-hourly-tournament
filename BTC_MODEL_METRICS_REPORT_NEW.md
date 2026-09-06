# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T01:08:56.301552+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 235 | 175 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 271 | 211 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 381 | 199 | 182 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 381 | 199 | 182 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 171 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 171 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 171 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T19:00:00+00:00 | 172 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 199 | 103 | 96 | 51.76% | 51.76% | 51.76% | 1.76 pp | 7 | 17 | 0.41 |
| BTC Market Hours | nn | NN | 199 | 101 | 98 | 50.75% | 50.75% | 50.75% | 0.75 pp | 3 | 16 | 0.19 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 8 | 0.12 |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| BTC Market Hours | transformer | Transformer | 199 | 97 | 102 | 48.74% | 48.74% | 48.74% | 1.26 pp | -5 | 16 | -0.31 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 17 | -0.41 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 199 | 95 | 104 | 47.74% | 47.74% | 47.74% | 2.26 pp | -9 | 17 | -0.53 |
| BTC Daily | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 199 | 93 | 106 | 46.73% | 46.73% | 46.73% | 3.27 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 16 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 199 | 87 | 112 | 43.72% | 43.72% | 43.72% | 6.28 pp | -25 | 17 | -1.47 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| BTC Hourly | transformer | Transformer | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 8 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 16 | -1.69 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| BTC Daily | nn | NN | 201 | 92 | 109 | 45.77% | 45.77% | 45.77% | 4.23 pp | -17 | 9 | -1.89 |
| BTC Market Hours | lstm | LSTM | 199 | 84 | 115 | 42.21% | 42.21% | 42.21% | 7.79 pp | -31 | 16 | -1.94 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 199 | 82 | 117 | 41.21% | 41.21% | 41.21% | 8.79 pp | -35 | 17 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 199 | 81 | 118 | 40.70% | 40.70% | 40.70% | 9.30 pp | -37 | 17 | -2.18 |
| Consolidated Hourly | transformer | Transformer | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Hourly | rf | RandomForest | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 201 | 85 | 116 | 42.29% | 42.29% | 42.29% | 7.71 pp | -31 | 9 | -3.44 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Hourly | nn | NN | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 8 | -3.62 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 201 | 77 | 124 | 38.31% | 38.31% | 38.31% | 11.69 pp | -47 | 9 | -5.22 |
| BTC Daily | xgb | XGBoost | 211 | 75 | 136 | 35.55% | 35.55% | 35.55% | 14.45 pp | -61 | 10 | -6.10 |
| BTC Hourly | lstm | LSTM | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 201 | 68 | 133 | 33.83% | 33.83% | 33.83% | 16.17 pp | -65 | 9 | -7.22 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 175 | 88 | 87 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 175 | 81 | 94 | 46.29% | 46.29% | 46.29% | 3.71 pp | -13 | 8 | -1.62 |
| BTC Hourly | rf | RandomForest | 175 | 74 | 101 | 42.29% | 42.29% | 42.29% | 7.71 pp | -27 | 8 | -3.38 |
| BTC Hourly | nn | NN | 175 | 73 | 102 | 41.71% | 41.71% | 41.71% | 8.29 pp | -29 | 8 | -3.62 |
| BTC Hourly | lstm | LSTM | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 8 | -6.12 |
| BTC Hourly | xgb | XGBoost | 175 | 63 | 112 | 36.00% | 36.00% | 36.00% | 14.00 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 201 | 97 | 104 | 48.26% | 48.26% | 48.26% | 1.74 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 201 | 92 | 109 | 45.77% | 45.77% | 45.77% | 4.23 pp | -17 | 9 | -1.89 |
| BTC Daily | transformer | Transformer | 201 | 85 | 116 | 42.29% | 42.29% | 42.29% | 7.71 pp | -31 | 9 | -3.44 |
| BTC Daily | rf | RandomForest | 201 | 77 | 124 | 38.31% | 38.31% | 38.31% | 11.69 pp | -47 | 9 | -5.22 |
| BTC Daily | xgb | XGBoost | 211 | 75 | 136 | 35.55% | 35.55% | 35.55% | 14.45 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 201 | 68 | 133 | 33.83% | 33.83% | 33.83% | 16.17 pp | -65 | 9 | -7.22 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 199 | 101 | 98 | 50.75% | 50.75% | 50.75% | 0.75 pp | 3 | 16 | 0.19 |
| BTC Market Hours | transformer | Transformer | 199 | 97 | 102 | 48.74% | 48.74% | 48.74% | 1.26 pp | -5 | 16 | -0.31 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 199 | 93 | 106 | 46.73% | 46.73% | 46.73% | 3.27 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 16 | -1.19 |
| BTC Market Hours | xgb | XGBoost | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 16 | -1.69 |
| BTC Market Hours | lstm | LSTM | 199 | 84 | 115 | 42.21% | 42.21% | 42.21% | 7.79 pp | -31 | 16 | -1.94 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 199 | 103 | 96 | 51.76% | 51.76% | 51.76% | 1.76 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 17 | -0.41 |
| BTC Market Hours Daily | nn | NN | 199 | 95 | 104 | 47.74% | 47.74% | 47.74% | 2.26 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 199 | 87 | 112 | 43.72% | 43.72% | 43.72% | 6.28 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 199 | 82 | 117 | 41.21% | 41.21% | 41.21% | 8.79 pp | -35 | 17 | -2.06 |
| BTC Market Hours Daily | lstm | LSTM | 199 | 81 | 118 | 40.70% | 40.70% | 40.70% | 9.30 pp | -37 | 17 | -2.18 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | transformer | Transformer | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 76 | 95 | 44.44% | 44.44% | 44.44% | 5.56 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 49 | 21 | 28 | 42.86% | 42.86% | 42.86% | 7.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
