# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T07:20:52.302419+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 105 | 45 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 141 | 81 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 147 | 69 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 147 | 69 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 15:00:00+00:00 | 54 | 54 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 15:00:00+00:00 | 54 | 54 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 15:00:00+00:00 | 54 | 1 | 53 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 15:00:00+00:00 | 54 | 1 | 53 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 6 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 6 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | transformer | Transformer | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Daily | transformer | Transformer | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 2 | -0.50 |
| BTC Market Hours | rf | RandomForest | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| BTC Hourly | nn | NN | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 2 | -1.50 |
| BTC Market Hours | transformer | Transformer | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours Daily | rf | RandomForest | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 7 | -2.14 |
| BTC Market Hours | lstm | LSTM | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 7 | -2.43 |
| Consolidated Hourly | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |
| BTC Hourly | lstm | LSTM | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 71 | 25 | 46 | 35.21% | 35.21% | 35.21% | 14.79 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 81 | 26 | 55 | 32.10% | 32.10% | 32.10% | 17.90 pp | -29 | 5 | -5.80 |
| BTC Hourly | rf | RandomForest | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 45 | 14 | 31 | 31.11% | 31.11% | 31.11% | 18.89 pp | -17 | 2 | -8.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 2 | -0.50 |
| BTC Hourly | nn | NN | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 2 | -1.50 |
| BTC Hourly | lstm | LSTM | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 2 | -3.50 |
| BTC Hourly | rf | RandomForest | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 45 | 14 | 31 | 31.11% | 31.11% | 31.11% | 18.89 pp | -17 | 2 | -8.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Daily | nn | NN | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Daily | transformer | Transformer | 71 | 35 | 36 | 49.30% | 49.30% | 49.30% | 0.70 pp | -1 | 4 | -0.25 |
| BTC Daily | rf | RandomForest | 71 | 28 | 43 | 39.44% | 39.44% | 39.44% | 10.56 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 71 | 25 | 46 | 35.21% | 35.21% | 35.21% | 14.79 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 81 | 26 | 55 | 32.10% | 32.10% | 32.10% | 17.90 pp | -29 | 5 | -5.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 6 | 1.50 |
| BTC Market Hours | rf | RandomForest | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 6 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| BTC Market Hours | lstm | LSTM | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 6 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | transformer | Transformer | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 7 | 0.14 |
| BTC Market Hours Daily | rf | RandomForest | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Market Hours Daily | nn | NN | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | xgb | XGBoost | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 7 | -2.14 |
| BTC Market Hours Daily | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 7 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 6 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 54 | 30 | 24 | 55.56% | 55.56% | 55.56% | 5.56 pp | 6 | 6 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
