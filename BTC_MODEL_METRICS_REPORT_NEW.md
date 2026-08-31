# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T09:19:15.223240+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 143 | 83 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 179 | 119 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 211 | 107 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 211 | 107 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 109 | 58 | 51 | 53.21% | 53.21% | 53.21% | 3.21 pp | 7 | 5 | 1.40 |
| BTC Hourly | nn | NN | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 4 | 1.25 |
| BTC Market Hours | nn | NN | 107 | 58 | 49 | 54.21% | 54.21% | 54.21% | 4.21 pp | 9 | 9 | 1.00 |
| BTC Daily | nn | NN | 109 | 56 | 53 | 51.38% | 51.38% | 51.38% | 1.38 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 4 | 0.25 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Market Hours Daily | nn | NN | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 9 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 10 | -2.10 |
| BTC Daily | rf | RandomForest | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 5 | -2.20 |
| BTC Hourly | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 4 | -2.25 |
| BTC Market Hours | transformer | Transformer | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Daily | transformer | Transformer | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Market Hours | lstm | LSTM | 107 | 36 | 71 | 33.64% | 33.64% | 33.64% | 16.36 pp | -35 | 9 | -3.89 |
| BTC Market Hours Daily | lstm | LSTM | 107 | 34 | 73 | 31.78% | 31.78% | 31.78% | 18.22 pp | -39 | 10 | -3.90 |
| BTC Daily | lstm | LSTM | 109 | 42 | 67 | 38.53% | 38.53% | 38.53% | 11.47 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 6 | -5.17 |
| BTC Hourly | xgb | XGBoost | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 4 | -5.25 |
| BTC Hourly | lstm | LSTM | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 4 | -6.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 4 | 1.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 83 | 42 | 41 | 50.60% | 50.60% | 50.60% | 0.60 pp | 1 | 4 | 0.25 |
| BTC Hourly | transformer | Transformer | 83 | 41 | 42 | 49.40% | 49.40% | 49.40% | 0.60 pp | -1 | 4 | -0.25 |
| BTC Hourly | rf | RandomForest | 83 | 37 | 46 | 44.58% | 44.58% | 44.58% | 5.42 pp | -9 | 4 | -2.25 |
| BTC Hourly | xgb | XGBoost | 83 | 31 | 52 | 37.35% | 37.35% | 37.35% | 12.65 pp | -21 | 4 | -5.25 |
| BTC Hourly | lstm | LSTM | 83 | 29 | 54 | 34.94% | 34.94% | 34.94% | 15.06 pp | -25 | 4 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 109 | 58 | 51 | 53.21% | 53.21% | 53.21% | 3.21 pp | 7 | 5 | 1.40 |
| BTC Daily | nn | NN | 109 | 56 | 53 | 51.38% | 51.38% | 51.38% | 1.38 pp | 3 | 5 | 0.60 |
| BTC Daily | rf | RandomForest | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 5 | -2.20 |
| BTC Daily | transformer | Transformer | 109 | 48 | 61 | 44.04% | 44.04% | 44.04% | 5.96 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 109 | 42 | 67 | 38.53% | 38.53% | 38.53% | 11.47 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 119 | 44 | 75 | 36.97% | 36.97% | 36.97% | 13.03 pp | -31 | 6 | -5.17 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 107 | 58 | 49 | 54.21% | 54.21% | 54.21% | 4.21 pp | 9 | 9 | 1.00 |
| BTC Market Hours | rf | RandomForest | 107 | 52 | 55 | 48.60% | 48.60% | 48.60% | 1.40 pp | -3 | 9 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 9 | -1.00 |
| BTC Market Hours | transformer | Transformer | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 9 | -2.33 |
| BTC Market Hours | lstm | LSTM | 107 | 36 | 71 | 33.64% | 33.64% | 33.64% | 16.36 pp | -35 | 9 | -3.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 107 | 50 | 57 | 46.73% | 46.73% | 46.73% | 3.27 pp | -7 | 10 | -0.70 |
| BTC Market Hours Daily | nn | NN | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 107 | 49 | 58 | 45.79% | 45.79% | 45.79% | 4.21 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | transformer | Transformer | 107 | 48 | 59 | 44.86% | 44.86% | 44.86% | 5.14 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | xgb | XGBoost | 107 | 43 | 64 | 40.19% | 40.19% | 40.19% | 9.81 pp | -21 | 10 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 107 | 34 | 73 | 31.78% | 31.78% | 31.78% | 18.22 pp | -39 | 10 | -3.90 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 1 | 2 | 33.33% | 33.33% | 33.33% | 16.67 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
