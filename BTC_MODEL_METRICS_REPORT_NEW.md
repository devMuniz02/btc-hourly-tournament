# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T18:44:16.205670+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 114 | 54 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 150 | 90 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 162 | 78 | 84 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 17:00:00+00:00 | 162 | 78 | 84 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 61 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 1 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 00:00:00+00:00 | 61 | 1 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 78 | 46 | 32 | 58.97% | 58.97% | 58.97% | 8.97 pp | 14 | 6 | 2.33 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 41 | 37 | 52.56% | 52.56% | 52.56% | 2.56 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | rf | RandomForest | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 7 | -0.29 |
| BTC Daily | transformer | Transformer | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 80 | 38 | 42 | 47.50% | 47.50% | 47.50% | 2.50 pp | -4 | 4 | -1.00 |
| BTC Daily | nn | NN | 80 | 38 | 42 | 47.50% | 47.50% | 47.50% | 2.50 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| BTC Hourly | nn | NN | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | nn | NN | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 7 | -2.29 |
| BTC Market Hours | xgb | XGBoost | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |
| BTC Hourly | lstm | LSTM | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 54 | 17 | 37 | 31.48% | 31.48% | 31.48% | 18.52 pp | -20 | 3 | -6.67 |
| BTC Daily | xgb | XGBoost | 90 | 27 | 63 | 30.00% | 30.00% | 30.00% | 20.00 pp | -36 | 5 | -7.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Hourly | transformer | Transformer | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 3 | -0.67 |
| BTC Hourly | nn | NN | 54 | 25 | 29 | 46.30% | 46.30% | 46.30% | 3.70 pp | -4 | 3 | -1.33 |
| BTC Hourly | lstm | LSTM | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 3 | -4.00 |
| BTC Hourly | rf | RandomForest | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 3 | -4.00 |
| BTC Hourly | xgb | XGBoost | 54 | 17 | 37 | 31.48% | 31.48% | 31.48% | 18.52 pp | -20 | 3 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 80 | 38 | 42 | 47.50% | 47.50% | 47.50% | 2.50 pp | -4 | 4 | -1.00 |
| BTC Daily | nn | NN | 80 | 38 | 42 | 47.50% | 47.50% | 47.50% | 2.50 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 80 | 31 | 49 | 38.75% | 38.75% | 38.75% | 11.25 pp | -18 | 4 | -4.50 |
| BTC Daily | lstm | LSTM | 80 | 28 | 52 | 35.00% | 35.00% | 35.00% | 15.00 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 90 | 27 | 63 | 30.00% | 30.00% | 30.00% | 20.00 pp | -36 | 5 | -7.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 78 | 46 | 32 | 58.97% | 58.97% | 58.97% | 8.97 pp | 14 | 6 | 2.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | rf | RandomForest | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | lstm | LSTM | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 78 | 33 | 45 | 42.31% | 42.31% | 42.31% | 7.69 pp | -12 | 6 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 6 | -2.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 78 | 41 | 37 | 52.56% | 52.56% | 52.56% | 2.56 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 78 | 38 | 40 | 48.72% | 48.72% | 48.72% | 1.28 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | rf | RandomForest | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | lstm | LSTM | 78 | 32 | 46 | 41.03% | 41.03% | 41.03% | 8.97 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 7 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 61 | 34 | 27 | 55.74% | 55.74% | 55.74% | 5.74 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 61 | 31 | 30 | 50.82% | 50.82% | 50.82% | 0.82 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 61 | 30 | 31 | 49.18% | 49.18% | 49.18% | 0.82 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 61 | 26 | 35 | 42.62% | 42.62% | 42.62% | 7.38 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 61 | 25 | 36 | 40.98% | 40.98% | 40.98% | 9.02 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 61 | 22 | 39 | 36.07% | 36.07% | 36.07% | 13.93 pp | -17 | 7 | -2.43 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
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
