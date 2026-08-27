# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T20:34:51.735038+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 18 | 84 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 114 | 54 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 19:00:00+00:00 | 102 | 42 | 60 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 19:00:00+00:00 | 102 | 42 | 60 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 14:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 14:00:00+00:00 | 31 | 31 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 14:00:00+00:00 | 31 | 1 | 30 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 14:00:00+00:00 | 31 | 1 | 30 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 42 | 24 | 18 | 57.14% | 57.14% | 57.14% | 7.14 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 22 | 20 | 52.38% | 52.38% | 52.38% | 2.38 pp | 2 | 5 | 0.40 |
| BTC Daily | nn | NN | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | xgb | XGBoost | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| BTC Market Hours | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| BTC Hourly | lstm | LSTM | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | nn | NN | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 4 | -2.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 4 | -2.75 |
| BTC Daily | xgb | XGBoost | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 4 | -3.50 |
| BTC Hourly | nn | NN | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 1 | -4.00 |
| BTC Market Hours | lstm | LSTM | 42 | 13 | 29 | 30.95% | 30.95% | 30.95% | 19.05 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 42 | 11 | 31 | 26.19% | 26.19% | 26.19% | 23.81 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 3 | -4.67 |
| BTC Hourly | xgb | XGBoost | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 18 | 9 | 9 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | lstm | LSTM | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 18 | 8 | 10 | 44.44% | 44.44% | 44.44% | 5.56 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 18 | 7 | 11 | 38.89% | 38.89% | 38.89% | 11.11 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 18 | 6 | 12 | 33.33% | 33.33% | 33.33% | 16.67 pp | -6 | 1 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 44 | 23 | 21 | 52.27% | 52.27% | 52.27% | 2.27 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 54 | 20 | 34 | 37.04% | 37.04% | 37.04% | 12.96 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 44 | 15 | 29 | 34.09% | 34.09% | 34.09% | 15.91 pp | -14 | 3 | -4.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 42 | 24 | 18 | 57.14% | 57.14% | 57.14% | 7.14 pp | 6 | 4 | 1.50 |
| BTC Market Hours | rf | RandomForest | 42 | 21 | 21 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | transformer | Transformer | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 4 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 4 | -1.50 |
| BTC Market Hours | lstm | LSTM | 42 | 13 | 29 | 30.95% | 30.95% | 30.95% | 19.05 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 42 | 22 | 20 | 52.38% | 52.38% | 52.38% | 2.38 pp | 2 | 5 | 0.40 |
| BTC Market Hours Daily | transformer | Transformer | 42 | 20 | 22 | 47.62% | 47.62% | 47.62% | 2.38 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | xgb | XGBoost | 42 | 19 | 23 | 45.24% | 45.24% | 45.24% | 4.76 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | nn | NN | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 42 | 18 | 24 | 42.86% | 42.86% | 42.86% | 7.14 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | lstm | LSTM | 42 | 11 | 31 | 26.19% | 26.19% | 26.19% | 23.81 pp | -20 | 5 | -4.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 4 | -2.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 31 | 18 | 13 | 58.06% | 58.06% | 58.06% | 8.06 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
