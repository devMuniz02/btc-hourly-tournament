# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T18:27:40.839900+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 17:00:00+00:00 | 253 | 130 | 123 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 17:00:00+00:00 | 253 | 130 | 123 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 106 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 23:00:00+00:00 | 106 | 13 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| BTC Market Hours | nn | NN | 130 | 69 | 61 | 53.08% | 53.08% | 53.08% | 3.08 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| BTC Daily | mlp_sklearn | MLPClassifier | 132 | 65 | 67 | 49.24% | 49.24% | 49.24% | 0.76 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 11 | -0.55 |
| BTC Market Hours | rf | RandomForest | 130 | 62 | 68 | 47.69% | 47.69% | 47.69% | 2.31 pp | -6 | 10 | -0.60 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 130 | 59 | 71 | 45.38% | 45.38% | 45.38% | 4.62 pp | -12 | 11 | -1.09 |
| BTC Hourly | nn | NN | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 11 | -1.27 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 10 | -1.40 |
| BTC Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 5 | -1.60 |
| BTC Daily | nn | NN | 132 | 61 | 71 | 46.21% | 46.21% | 46.21% | 3.79 pp | -10 | 6 | -1.67 |
| BTC Market Hours | transformer | Transformer | 130 | 55 | 75 | 42.31% | 42.31% | 42.31% | 7.69 pp | -20 | 10 | -2.00 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 10 | -2.20 |
| BTC Daily | transformer | Transformer | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 130 | 52 | 78 | 40.00% | 40.00% | 40.00% | 10.00 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 11 | -2.73 |
| BTC Market Hours | lstm | LSTM | 130 | 50 | 80 | 38.46% | 38.46% | 38.46% | 11.54 pp | -30 | 10 | -3.00 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 132 | 54 | 78 | 40.91% | 40.91% | 40.91% | 9.09 pp | -24 | 6 | -4.00 |
| BTC Hourly | rf | RandomForest | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 5 | -4.00 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Daily | xgb | XGBoost | 142 | 51 | 91 | 35.92% | 35.92% | 35.92% | 14.08 pp | -40 | 7 | -5.71 |
| BTC Hourly | xgb | XGBoost | 106 | 37 | 69 | 34.91% | 34.91% | 34.91% | 15.09 pp | -32 | 5 | -6.40 |
| BTC Daily | lstm | LSTM | 132 | 45 | 87 | 34.09% | 34.09% | 34.09% | 15.91 pp | -42 | 6 | -7.00 |
| BTC Hourly | lstm | LSTM | 106 | 33 | 73 | 31.13% | 31.13% | 31.13% | 18.87 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 5 | -0.80 |
| BTC Hourly | nn | NN | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 5 | -1.60 |
| BTC Hourly | rf | RandomForest | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 5 | -4.00 |
| BTC Hourly | xgb | XGBoost | 106 | 37 | 69 | 34.91% | 34.91% | 34.91% | 15.09 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 106 | 33 | 73 | 31.13% | 31.13% | 31.13% | 18.87 pp | -40 | 5 | -8.00 |

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
| Consolidated Hourly | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Hourly | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 106 | 55 | 51 | 51.89% | 51.89% | 51.89% | 1.89 pp | 4 | 9 | 0.44 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 106 | 54 | 52 | 50.94% | 50.94% | 50.94% | 0.94 pp | 2 | 9 | 0.22 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 9 | -0.89 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 9 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 106 | 44 | 62 | 41.51% | 41.51% | 41.51% | 8.49 pp | -18 | 9 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 13 | 8 | 5 | 61.54% | 61.54% | 61.54% | 11.54 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
