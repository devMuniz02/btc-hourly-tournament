# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T12:29:57.791539+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 145 | 85 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 181 | 121 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 213 | 109 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 213 | 109 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T13:00:00+00:00 | 87 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T13:00:00+00:00 | 87 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T13:00:00+00:00 | 87 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T13:00:00+00:00 | 88 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 109 | 58 | 51 | 53.21% | 53.21% | 53.21% | 3.21 pp | 7 | 9 | 0.78 |
| BTC Hourly | transformer | Transformer | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 2 | 2 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 111 | 55 | 56 | 49.55% | 49.55% | 49.55% | 0.45 pp | -1 | 5 | -0.20 |
| BTC Hourly | nn | NN | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 109 | 50 | 59 | 45.87% | 45.87% | 45.87% | 4.13 pp | -9 | 10 | -0.90 |
| BTC Daily | nn | NN | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 109 | 47 | 62 | 43.12% | 43.12% | 43.12% | 6.88 pp | -15 | 10 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | transformer | Transformer | 109 | 45 | 64 | 41.28% | 41.28% | 41.28% | 8.72 pp | -19 | 9 | -2.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 109 | 42 | 67 | 38.53% | 38.53% | 38.53% | 11.47 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 109 | 43 | 66 | 39.45% | 39.45% | 39.45% | 10.55 pp | -23 | 9 | -2.56 |
| BTC Market Hours | xgb | XGBoost | 109 | 43 | 66 | 39.45% | 39.45% | 39.45% | 10.55 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 5 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 109 | 41 | 68 | 37.61% | 37.61% | 37.61% | 12.39 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 5 | -3.80 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 0 | 4 | 0.00% | 0.00% | 0.00% | 50.00 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 121 | 44 | 77 | 36.36% | 36.36% | 36.36% | 13.64 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 111 | 39 | 72 | 35.14% | 35.14% | 35.14% | 14.86 pp | -33 | 5 | -6.60 |
| BTC Hourly | xgb | XGBoost | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 4 | -6.75 |
| BTC Hourly | lstm | LSTM | 85 | 28 | 57 | 32.94% | 32.94% | 32.94% | 17.06 pp | -29 | 4 | -7.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 85 | 42 | 43 | 49.41% | 49.41% | 49.41% | 0.59 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 85 | 38 | 47 | 44.71% | 44.71% | 44.71% | 5.29 pp | -9 | 4 | -2.25 |
| BTC Hourly | rf | RandomForest | 85 | 35 | 50 | 41.18% | 41.18% | 41.18% | 8.82 pp | -15 | 4 | -3.75 |
| BTC Hourly | xgb | XGBoost | 85 | 29 | 56 | 34.12% | 34.12% | 34.12% | 15.88 pp | -27 | 4 | -6.75 |
| BTC Hourly | lstm | LSTM | 85 | 28 | 57 | 32.94% | 32.94% | 32.94% | 17.06 pp | -29 | 4 | -7.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 111 | 55 | 56 | 49.55% | 49.55% | 49.55% | 0.45 pp | -1 | 5 | -0.20 |
| BTC Daily | nn | NN | 111 | 53 | 58 | 47.75% | 47.75% | 47.75% | 2.25 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 111 | 49 | 62 | 44.14% | 44.14% | 44.14% | 5.86 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 111 | 46 | 65 | 41.44% | 41.44% | 41.44% | 8.56 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 121 | 44 | 77 | 36.36% | 36.36% | 36.36% | 13.64 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 111 | 39 | 72 | 35.14% | 35.14% | 35.14% | 14.86 pp | -33 | 5 | -6.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 109 | 58 | 51 | 53.21% | 53.21% | 53.21% | 3.21 pp | 7 | 9 | 0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 9 | -0.56 |
| BTC Market Hours | transformer | Transformer | 109 | 45 | 64 | 41.28% | 41.28% | 41.28% | 8.72 pp | -19 | 9 | -2.11 |
| BTC Market Hours | lstm | LSTM | 109 | 43 | 66 | 39.45% | 39.45% | 39.45% | 10.55 pp | -23 | 9 | -2.56 |
| BTC Market Hours | xgb | XGBoost | 109 | 43 | 66 | 39.45% | 39.45% | 39.45% | 10.55 pp | -23 | 9 | -2.56 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 109 | 55 | 54 | 50.46% | 50.46% | 50.46% | 0.46 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 109 | 50 | 59 | 45.87% | 45.87% | 45.87% | 4.13 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 109 | 47 | 62 | 43.12% | 43.12% | 43.12% | 6.88 pp | -15 | 10 | -1.50 |
| BTC Market Hours Daily | xgb | XGBoost | 109 | 42 | 67 | 38.53% | 38.53% | 38.53% | 11.47 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 109 | 41 | 68 | 37.61% | 37.61% | 37.61% | 12.39 pp | -27 | 10 | -2.70 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | xgb | XGBoost | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 87 | 44 | 43 | 50.57% | 50.57% | 50.57% | 0.57 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 87 | 41 | 46 | 47.13% | 47.13% | 47.13% | 2.87 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 4 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 2 | 2 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 0 | 4 | 0.00% | 0.00% | 0.00% | 50.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
