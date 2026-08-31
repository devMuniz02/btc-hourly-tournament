# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T05:07:44.321150+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 140 | 80 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 176 | 116 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 208 | 104 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 208 | 104 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 4 | 1.50 |
| BTC Market Hours | nn | NN | 104 | 57 | 47 | 54.81% | 54.81% | 54.81% | 4.81 pp | 10 | 8 | 1.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 106 | 56 | 50 | 52.83% | 52.83% | 52.83% | 2.83 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| BTC Daily | nn | NN | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | rf | RandomForest | 104 | 51 | 53 | 49.04% | 49.04% | 49.04% | 0.96 pp | -2 | 8 | -0.25 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| BTC Market Hours Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 104 | 46 | 58 | 44.23% | 44.23% | 44.23% | 5.77 pp | -12 | 9 | -1.33 |
| BTC Hourly | rf | RandomForest | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| BTC Daily | rf | RandomForest | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 9 | -2.44 |
| BTC Market Hours | transformer | Transformer | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 8 | -2.75 |
| BTC Market Hours | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 8 | -2.75 |
| BTC Market Hours Daily | lstm | LSTM | 104 | 33 | 71 | 31.73% | 31.73% | 31.73% | 18.27 pp | -38 | 9 | -4.22 |
| BTC Market Hours | lstm | LSTM | 104 | 35 | 69 | 33.65% | 33.65% | 33.65% | 16.35 pp | -34 | 8 | -4.25 |
| BTC Daily | lstm | LSTM | 106 | 42 | 64 | 39.62% | 39.62% | 39.62% | 10.38 pp | -22 | 5 | -4.40 |
| BTC Hourly | xgb | XGBoost | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 116 | 43 | 73 | 37.07% | 37.07% | 37.07% | 12.93 pp | -30 | 6 | -5.00 |
| BTC Hourly | lstm | LSTM | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 4 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 4 | 1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 4 | 1.00 |
| BTC Hourly | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | rf | RandomForest | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 4 | -1.50 |
| BTC Hourly | xgb | XGBoost | 80 | 30 | 50 | 37.50% | 37.50% | 37.50% | 12.50 pp | -20 | 4 | -5.00 |
| BTC Hourly | lstm | LSTM | 80 | 29 | 51 | 36.25% | 36.25% | 36.25% | 13.75 pp | -22 | 4 | -5.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 106 | 56 | 50 | 52.83% | 52.83% | 52.83% | 2.83 pp | 6 | 5 | 1.20 |
| BTC Daily | nn | NN | 106 | 53 | 53 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Daily | rf | RandomForest | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 106 | 42 | 64 | 39.62% | 39.62% | 39.62% | 10.38 pp | -22 | 5 | -4.40 |
| BTC Daily | xgb | XGBoost | 116 | 43 | 73 | 37.07% | 37.07% | 37.07% | 12.93 pp | -30 | 6 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 104 | 57 | 47 | 54.81% | 54.81% | 54.81% | 4.81 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 104 | 51 | 53 | 49.04% | 49.04% | 49.04% | 0.96 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 8 | -1.00 |
| BTC Market Hours | transformer | Transformer | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 8 | -2.75 |
| BTC Market Hours | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 8 | -2.75 |
| BTC Market Hours | lstm | LSTM | 104 | 35 | 69 | 33.65% | 33.65% | 33.65% | 16.35 pp | -34 | 8 | -4.25 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 9 | -0.67 |
| BTC Market Hours Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 104 | 46 | 58 | 44.23% | 44.23% | 44.23% | 5.77 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | xgb | XGBoost | 104 | 41 | 63 | 39.42% | 39.42% | 39.42% | 10.58 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | lstm | LSTM | 104 | 33 | 71 | 31.73% | 31.73% | 31.73% | 18.27 pp | -38 | 9 | -4.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
