# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T19:26:56.121996+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 150 | 90 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 186 | 126 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 18:00:00+00:00 | 225 | 114 | 111 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 18:00:00+00:00 | 224 | 113 | 111 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 91 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 15:00:00+00:00 | 91 | 5 | 86 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| BTC Market Hours | nn | NN | 114 | 62 | 52 | 54.39% | 54.39% | 54.39% | 4.39 pp | 10 | 9 | 1.11 |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 113 | 57 | 56 | 50.44% | 50.44% | 50.44% | 0.44 pp | 1 | 10 | 0.10 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 90 | 44 | 46 | 48.89% | 48.89% | 48.89% | 1.11 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 6 | -0.67 |
| BTC Market Hours | rf | RandomForest | 114 | 54 | 60 | 47.37% | 47.37% | 47.37% | 2.63 pp | -6 | 9 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Hourly | nn | NN | 90 | 43 | 47 | 47.78% | 47.78% | 47.78% | 2.22 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Daily | nn | NN | 116 | 54 | 62 | 46.55% | 46.55% | 46.55% | 3.45 pp | -8 | 6 | -1.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 4 | -1.50 |
| BTC Daily | transformer | Transformer | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 6 | -2.00 |
| BTC Market Hours | transformer | Transformer | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 9 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 114 | 46 | 68 | 40.35% | 40.35% | 40.35% | 9.65 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 10 | -2.50 |
| BTC Market Hours | lstm | LSTM | 114 | 44 | 70 | 38.60% | 38.60% | 38.60% | 11.40 pp | -26 | 9 | -2.89 |
| BTC Market Hours Daily | lstm | LSTM | 113 | 42 | 71 | 37.17% | 37.17% | 37.17% | 12.83 pp | -29 | 10 | -2.90 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 116 | 46 | 70 | 39.66% | 39.66% | 39.66% | 10.34 pp | -24 | 6 | -4.00 |
| BTC Hourly | rf | RandomForest | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 4 | -4.50 |
| BTC Daily | xgb | XGBoost | 126 | 45 | 81 | 35.71% | 35.71% | 35.71% | 14.29 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 116 | 40 | 76 | 34.48% | 34.48% | 34.48% | 15.52 pp | -36 | 6 | -6.00 |
| BTC Hourly | xgb | XGBoost | 90 | 30 | 60 | 33.33% | 33.33% | 33.33% | 16.67 pp | -30 | 4 | -7.50 |
| BTC Hourly | lstm | LSTM | 90 | 29 | 61 | 32.22% | 32.22% | 32.22% | 17.78 pp | -32 | 4 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 90 | 44 | 46 | 48.89% | 48.89% | 48.89% | 1.11 pp | -2 | 4 | -0.50 |
| BTC Hourly | nn | NN | 90 | 43 | 47 | 47.78% | 47.78% | 47.78% | 2.22 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 90 | 36 | 54 | 40.00% | 40.00% | 40.00% | 10.00 pp | -18 | 4 | -4.50 |
| BTC Hourly | xgb | XGBoost | 90 | 30 | 60 | 33.33% | 33.33% | 33.33% | 16.67 pp | -30 | 4 | -7.50 |
| BTC Hourly | lstm | LSTM | 90 | 29 | 61 | 32.22% | 32.22% | 32.22% | 17.78 pp | -32 | 4 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 116 | 56 | 60 | 48.28% | 48.28% | 48.28% | 1.72 pp | -4 | 6 | -0.67 |
| BTC Daily | nn | NN | 116 | 54 | 62 | 46.55% | 46.55% | 46.55% | 3.45 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 116 | 52 | 64 | 44.83% | 44.83% | 44.83% | 5.17 pp | -12 | 6 | -2.00 |
| BTC Daily | rf | RandomForest | 116 | 46 | 70 | 39.66% | 39.66% | 39.66% | 10.34 pp | -24 | 6 | -4.00 |
| BTC Daily | xgb | XGBoost | 126 | 45 | 81 | 35.71% | 35.71% | 35.71% | 14.29 pp | -36 | 7 | -5.14 |
| BTC Daily | lstm | LSTM | 116 | 40 | 76 | 34.48% | 34.48% | 34.48% | 15.52 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 114 | 62 | 52 | 54.39% | 54.39% | 54.39% | 4.39 pp | 10 | 9 | 1.11 |
| BTC Market Hours | rf | RandomForest | 114 | 54 | 60 | 47.37% | 47.37% | 47.37% | 2.63 pp | -6 | 9 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 114 | 53 | 61 | 46.49% | 46.49% | 46.49% | 3.51 pp | -8 | 9 | -0.89 |
| BTC Market Hours | transformer | Transformer | 114 | 48 | 66 | 42.11% | 42.11% | 42.11% | 7.89 pp | -18 | 9 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 114 | 46 | 68 | 40.35% | 40.35% | 40.35% | 9.65 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 114 | 44 | 70 | 38.60% | 38.60% | 38.60% | 11.40 pp | -26 | 9 | -2.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 113 | 57 | 56 | 50.44% | 50.44% | 50.44% | 0.44 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 113 | 52 | 61 | 46.02% | 46.02% | 46.02% | 3.98 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | nn | NN | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 113 | 50 | 63 | 44.25% | 44.25% | 44.25% | 5.75 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 113 | 44 | 69 | 38.94% | 38.94% | 38.94% | 11.06 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 113 | 42 | 71 | 37.17% | 37.17% | 37.17% | 12.83 pp | -29 | 10 | -2.90 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 91 | 50 | 41 | 54.95% | 54.95% | 54.95% | 4.95 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 5 | 4 | 1 | 80.00% | 80.00% | 80.00% | 30.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 5 | 2 | 3 | 40.00% | 40.00% | 40.00% | 10.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 5 | 1 | 4 | 20.00% | 20.00% | 20.00% | 30.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
