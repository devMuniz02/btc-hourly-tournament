# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T14:34:39.121157+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 146 | 86 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 182 | 122 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 13:00:00+00:00 | 216 | 110 | 106 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 13:00:00+00:00 | 216 | 110 | 106 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 89 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 14:00:00+00:00 | 89 | 4 | 85 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| BTC Market Hours | nn | NN | 110 | 59 | 51 | 53.64% | 53.64% | 53.64% | 3.64 pp | 8 | 9 | 0.89 |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| BTC Hourly | transformer | Transformer | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 112 | 56 | 56 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 9 | -0.44 |
| BTC Market Hours | rf | RandomForest | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 9 | -0.44 |
| BTC Hourly | nn | NN | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 110 | 48 | 62 | 43.64% | 43.64% | 43.64% | 6.36 pp | -14 | 10 | -1.40 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 4 | -1.50 |
| BTC Market Hours | transformer | Transformer | 110 | 46 | 64 | 41.82% | 41.82% | 41.82% | 8.18 pp | -18 | 9 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | lstm | LSTM | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 9 | -2.44 |
| BTC Market Hours | xgb | XGBoost | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 110 | 42 | 68 | 38.18% | 38.18% | 38.18% | 11.82 pp | -26 | 10 | -2.60 |
| BTC Daily | transformer | Transformer | 112 | 49 | 63 | 43.75% | 43.75% | 43.75% | 6.25 pp | -14 | 5 | -2.80 |
| BTC Market Hours Daily | lstm | LSTM | 110 | 41 | 69 | 37.27% | 37.27% | 37.27% | 12.73 pp | -28 | 10 | -2.80 |
| BTC Daily | rf | RandomForest | 112 | 46 | 66 | 41.07% | 41.07% | 41.07% | 8.93 pp | -20 | 5 | -4.00 |
| BTC Hourly | rf | RandomForest | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 4 | -4.00 |
| BTC Daily | xgb | XGBoost | 122 | 45 | 77 | 36.89% | 36.89% | 36.89% | 13.11 pp | -32 | 6 | -5.33 |
| BTC Daily | lstm | LSTM | 112 | 39 | 73 | 34.82% | 34.82% | 34.82% | 15.18 pp | -34 | 5 | -6.80 |
| BTC Hourly | xgb | XGBoost | 86 | 29 | 57 | 33.72% | 33.72% | 33.72% | 16.28 pp | -28 | 4 | -7.00 |
| BTC Hourly | lstm | LSTM | 86 | 28 | 58 | 32.56% | 32.56% | 32.56% | 17.44 pp | -30 | 4 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 86 | 44 | 42 | 51.16% | 51.16% | 51.16% | 1.16 pp | 2 | 4 | 0.50 |
| BTC Hourly | nn | NN | 86 | 42 | 44 | 48.84% | 48.84% | 48.84% | 1.16 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 86 | 40 | 46 | 46.51% | 46.51% | 46.51% | 3.49 pp | -6 | 4 | -1.50 |
| BTC Hourly | rf | RandomForest | 86 | 35 | 51 | 40.70% | 40.70% | 40.70% | 9.30 pp | -16 | 4 | -4.00 |
| BTC Hourly | xgb | XGBoost | 86 | 29 | 57 | 33.72% | 33.72% | 33.72% | 16.28 pp | -28 | 4 | -7.00 |
| BTC Hourly | lstm | LSTM | 86 | 28 | 58 | 32.56% | 32.56% | 32.56% | 17.44 pp | -30 | 4 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 112 | 56 | 56 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Daily | nn | NN | 112 | 54 | 58 | 48.21% | 48.21% | 48.21% | 1.79 pp | -4 | 5 | -0.80 |
| BTC Daily | transformer | Transformer | 112 | 49 | 63 | 43.75% | 43.75% | 43.75% | 6.25 pp | -14 | 5 | -2.80 |
| BTC Daily | rf | RandomForest | 112 | 46 | 66 | 41.07% | 41.07% | 41.07% | 8.93 pp | -20 | 5 | -4.00 |
| BTC Daily | xgb | XGBoost | 122 | 45 | 77 | 36.89% | 36.89% | 36.89% | 13.11 pp | -32 | 6 | -5.33 |
| BTC Daily | lstm | LSTM | 112 | 39 | 73 | 34.82% | 34.82% | 34.82% | 15.18 pp | -34 | 5 | -6.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 110 | 59 | 51 | 53.64% | 53.64% | 53.64% | 3.64 pp | 8 | 9 | 0.89 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 9 | -0.44 |
| BTC Market Hours | rf | RandomForest | 110 | 53 | 57 | 48.18% | 48.18% | 48.18% | 1.82 pp | -4 | 9 | -0.44 |
| BTC Market Hours | transformer | Transformer | 110 | 46 | 64 | 41.82% | 41.82% | 41.82% | 8.18 pp | -18 | 9 | -2.00 |
| BTC Market Hours | lstm | LSTM | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 9 | -2.44 |
| BTC Market Hours | xgb | XGBoost | 110 | 44 | 66 | 40.00% | 40.00% | 40.00% | 10.00 pp | -22 | 9 | -2.44 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 110 | 56 | 54 | 50.91% | 50.91% | 50.91% | 0.91 pp | 2 | 10 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 110 | 50 | 60 | 45.45% | 45.45% | 45.45% | 4.55 pp | -10 | 10 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 110 | 49 | 61 | 44.55% | 44.55% | 44.55% | 5.45 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | nn | NN | 110 | 48 | 62 | 43.64% | 43.64% | 43.64% | 6.36 pp | -14 | 10 | -1.40 |
| BTC Market Hours Daily | xgb | XGBoost | 110 | 42 | 68 | 38.18% | 38.18% | 38.18% | 11.82 pp | -26 | 10 | -2.60 |
| BTC Market Hours Daily | lstm | LSTM | 110 | 41 | 69 | 37.27% | 37.27% | 37.27% | 12.73 pp | -28 | 10 | -2.80 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 89 | 48 | 41 | 53.93% | 53.93% | 53.93% | 3.93 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 89 | 38 | 51 | 42.70% | 42.70% | 42.70% | 7.30 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 4 | 3 | 1 | 75.00% | 75.00% | 75.00% | 25.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 4 | 1 | 3 | 25.00% | 25.00% | 25.00% | 25.00 pp | -2 | 1 | -2.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
