# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T19:35:00.413285+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 166 | 106 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 202 | 142 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 18:00:00+00:00 | 254 | 130 | 124 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 18:00:00+00:00 | 254 | 130 | 124 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 107 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 00:00:00+00:00 | 107 | 14 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| BTC Market Hours | nn | NN | 130 | 69 | 61 | 53.08% | 53.08% | 53.08% | 3.08 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| BTC Daily | mlp_sklearn | MLPClassifier | 132 | 65 | 67 | 49.24% | 49.24% | 49.24% | 0.76 pp | -2 | 6 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 106 | 52 | 54 | 49.06% | 49.06% | 49.06% | 0.94 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 11 | -0.55 |
| BTC Market Hours | rf | RandomForest | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Hourly | nn | NN | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | nn | NN | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Hourly | transformer | Transformer | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 11 | -1.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 10 | -1.40 |
| BTC Daily | nn | NN | 132 | 61 | 71 | 46.21% | 46.21% | 46.21% | 3.79 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |
| BTC Market Hours | transformer | Transformer | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 10 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 130 | 52 | 78 | 40.00% | 40.00% | 40.00% | 10.00 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 11 | -2.73 |
| BTC Market Hours | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 10 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |
| BTC Daily | rf | RandomForest | 132 | 54 | 78 | 40.91% | 40.91% | 40.91% | 9.09 pp | -24 | 6 | -4.00 |
| BTC Hourly | rf | RandomForest | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 5 | -4.00 |
| BTC Daily | xgb | XGBoost | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 106 | 38 | 68 | 35.85% | 35.85% | 35.85% | 14.15 pp | -30 | 5 | -6.00 |
| BTC Daily | lstm | LSTM | 132 | 45 | 87 | 34.09% | 34.09% | 34.09% | 15.91 pp | -42 | 6 | -7.00 |
| BTC Hourly | lstm | LSTM | 106 | 34 | 72 | 32.08% | 32.08% | 32.08% | 17.92 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 106 | 52 | 54 | 49.06% | 49.06% | 49.06% | 0.94 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 5 | -0.80 |
| BTC Hourly | transformer | Transformer | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 106 | 38 | 68 | 35.85% | 35.85% | 35.85% | 14.15 pp | -30 | 5 | -6.00 |
| BTC Hourly | lstm | LSTM | 106 | 34 | 72 | 32.08% | 32.08% | 32.08% | 17.92 pp | -38 | 5 | -7.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 132 | 65 | 67 | 49.24% | 49.24% | 49.24% | 0.76 pp | -2 | 6 | -0.33 |
| BTC Daily | nn | NN | 132 | 61 | 71 | 46.21% | 46.21% | 46.21% | 3.79 pp | -10 | 6 | -1.67 |
| BTC Daily | transformer | Transformer | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 6 | -2.33 |
| BTC Daily | rf | RandomForest | 132 | 54 | 78 | 40.91% | 40.91% | 40.91% | 9.09 pp | -24 | 6 | -4.00 |
| BTC Daily | xgb | XGBoost | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 7 | -5.71 |
| BTC Daily | lstm | LSTM | 132 | 45 | 87 | 34.09% | 34.09% | 34.09% | 15.91 pp | -42 | 6 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 130 | 69 | 61 | 53.08% | 53.08% | 53.08% | 3.08 pp | 8 | 10 | 0.80 |
| BTC Market Hours | rf | RandomForest | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 10 | -1.40 |
| BTC Market Hours | transformer | Transformer | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 10 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 10 | -2.20 |
| BTC Market Hours | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 10 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 11 | -0.55 |
| BTC Market Hours Daily | nn | NN | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | rf | RandomForest | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 130 | 52 | 78 | 40.00% | 40.00% | 40.00% | 10.00 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 11 | -2.73 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 107 | 56 | 51 | 52.34% | 52.34% | 52.34% | 2.34 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 107 | 54 | 53 | 50.47% | 50.47% | 50.47% | 0.47 pp | 1 | 10 | 0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 107 | 44 | 63 | 41.12% | 41.12% | 41.12% | 8.88 pp | -19 | 10 | -1.90 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 14 | 9 | 5 | 64.29% | 64.29% | 64.29% | 14.29 pp | 4 | 2 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 2 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 2 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
