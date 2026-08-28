# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T02:01:31.827698+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 22 | 80 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 118 | 58 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 111 | 46 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 111 | 46 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 1 | 34 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 18:00:00+00:00 | 35 | 1 | 34 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 1 | 2.00 |
| BTC Daily | transformer | Transformer | 48 | 26 | 22 | 54.17% | 54.17% | 54.17% | 4.17 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 46 | 25 | 21 | 54.35% | 54.35% | 54.35% | 4.35 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | lstm | LSTM | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| BTC Daily | xgb | XGBoost | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 4 | -3.50 |
| Consolidated Hourly | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |
| BTC Hourly | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 1 | -4.00 |
| BTC Market Hours | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 46 | 12 | 34 | 26.09% | 26.09% | 26.09% | 23.91 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 3 | -5.33 |
| BTC Hourly | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 1 | 2.00 |
| BTC Hourly | lstm | LSTM | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | transformer | Transformer | 22 | 9 | 13 | 40.91% | 40.91% | 40.91% | 9.09 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 48 | 26 | 22 | 54.17% | 54.17% | 54.17% | 4.17 pp | 4 | 3 | 1.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | nn | NN | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | rf | RandomForest | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 3 | -1.33 |
| BTC Daily | xgb | XGBoost | 58 | 22 | 36 | 37.93% | 37.93% | 37.93% | 12.07 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 3 | -5.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 46 | 25 | 21 | 54.35% | 54.35% | 54.35% | 4.35 pp | 4 | 4 | 1.00 |
| BTC Market Hours | rf | RandomForest | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| BTC Market Hours | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 46 | 12 | 34 | 26.09% | 26.09% | 26.09% | 23.91 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Hourly | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Hourly | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 4 | 0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 13 | 22 | 37.14% | 37.14% | 37.14% | 12.86 pp | -9 | 4 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 10 | 25 | 28.57% | 28.57% | 28.57% | 21.43 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
