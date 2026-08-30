# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T23:38:27.814055+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 137 | 77 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 173 | 113 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 203 | 101 | 102 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 22:00:00+00:00 | 202 | 100 | 102 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 80 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 80 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 0 | 80 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 09:00:00+00:00 | 80 | 0 | 80 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 8 | 1.12 |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| BTC Hourly | transformer | Transformer | 77 | 40 | 37 | 51.95% | 51.95% | 51.95% | 1.95 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 100 | 51 | 49 | 51.00% | 51.00% | 51.00% | 1.00 pp | 2 | 9 | 0.22 |
| BTC Hourly | nn | NN | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Daily | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 100 | 44 | 56 | 44.00% | 44.00% | 44.00% | 6.00 pp | -12 | 9 | -1.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 4 | -1.75 |
| BTC Daily | nn | NN | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | lstm | LSTM | 101 | 42 | 59 | 41.58% | 41.58% | 41.58% | 8.42 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 9 | -2.22 |
| BTC Market Hours | transformer | Transformer | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | xgb | XGBoost | 100 | 37 | 63 | 37.00% | 37.00% | 37.00% | 13.00 pp | -26 | 9 | -2.89 |
| BTC Market Hours | xgb | XGBoost | 101 | 38 | 63 | 37.62% | 37.62% | 37.62% | 12.38 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 5 | -5.40 |
| BTC Hourly | xgb | XGBoost | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Daily | xgb | XGBoost | 113 | 39 | 74 | 34.51% | 34.51% | 34.51% | 15.49 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 4 | -6.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 77 | 40 | 37 | 51.95% | 51.95% | 51.95% | 1.95 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 4 | -3.25 |
| BTC Hourly | xgb | XGBoost | 77 | 27 | 50 | 35.06% | 35.06% | 35.06% | 14.94 pp | -23 | 4 | -5.75 |
| BTC Hourly | lstm | LSTM | 77 | 26 | 51 | 33.77% | 33.77% | 33.77% | 16.23 pp | -25 | 4 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 103 | 47 | 56 | 45.63% | 45.63% | 45.63% | 4.37 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 113 | 39 | 74 | 34.51% | 34.51% | 34.51% | 15.49 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 101 | 55 | 46 | 54.46% | 54.46% | 54.46% | 4.46 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 101 | 49 | 52 | 48.51% | 48.51% | 48.51% | 1.49 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 8 | -0.62 |
| BTC Market Hours | lstm | LSTM | 101 | 42 | 59 | 41.58% | 41.58% | 41.58% | 8.42 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 101 | 38 | 63 | 37.62% | 37.62% | 37.62% | 12.38 pp | -25 | 8 | -3.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 100 | 51 | 49 | 51.00% | 51.00% | 51.00% | 1.00 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 100 | 46 | 54 | 46.00% | 46.00% | 46.00% | 4.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 100 | 44 | 56 | 44.00% | 44.00% | 44.00% | 6.00 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | lstm | LSTM | 100 | 40 | 60 | 40.00% | 40.00% | 40.00% | 10.00 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 100 | 37 | 63 | 37.00% | 37.00% | 37.00% | 13.00 pp | -26 | 9 | -2.89 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 80 | 44 | 36 | 55.00% | 55.00% | 55.00% | 5.00 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 80 | 43 | 37 | 53.75% | 53.75% | 53.75% | 3.75 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 80 | 37 | 43 | 46.25% | 46.25% | 46.25% | 3.75 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
