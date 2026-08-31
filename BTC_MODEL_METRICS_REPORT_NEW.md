# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T03:30:15.358502+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 175 | 115 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 207 | 103 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 207 | 103 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 103 | 56 | 47 | 54.37% | 54.37% | 54.37% | 4.37 pp | 9 | 8 | 1.12 |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| BTC Hourly | transformer | Transformer | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| BTC Market Hours | rf | RandomForest | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 8 | -0.38 |
| BTC Hourly | nn | NN | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| BTC Daily | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 5 | -1.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| BTC Market Hours | lstm | LSTM | 103 | 42 | 61 | 40.78% | 40.78% | 40.78% | 9.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | lstm | LSTM | 103 | 40 | 63 | 38.83% | 38.83% | 38.83% | 11.17 pp | -23 | 9 | -2.56 |
| BTC Daily | transformer | Transformer | 105 | 46 | 59 | 43.81% | 43.81% | 43.81% | 6.19 pp | -13 | 5 | -2.60 |
| BTC Market Hours | transformer | Transformer | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | xgb | XGBoost | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 9 | -3.00 |
| BTC Market Hours | xgb | XGBoost | 103 | 39 | 64 | 37.86% | 37.86% | 37.86% | 12.14 pp | -25 | 8 | -3.12 |
| BTC Hourly | rf | RandomForest | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 105 | 43 | 62 | 40.95% | 40.95% | 40.95% | 9.05 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 115 | 41 | 74 | 35.65% | 35.65% | 35.65% | 14.35 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 105 | 38 | 67 | 36.19% | 36.19% | 36.19% | 13.81 pp | -29 | 5 | -5.80 |
| BTC Hourly | lstm | LSTM | 80 | 27 | 53 | 33.75% | 33.75% | 33.75% | 16.25 pp | -26 | 4 | -6.50 |
| BTC Hourly | xgb | XGBoost | 80 | 27 | 53 | 33.75% | 33.75% | 33.75% | 16.25 pp | -26 | 4 | -6.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 80 | 41 | 39 | 51.25% | 51.25% | 51.25% | 1.25 pp | 2 | 4 | 0.50 |
| BTC Hourly | nn | NN | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 4 | -3.50 |
| BTC Hourly | lstm | LSTM | 80 | 27 | 53 | 33.75% | 33.75% | 33.75% | 16.25 pp | -26 | 4 | -6.50 |
| BTC Hourly | xgb | XGBoost | 80 | 27 | 53 | 33.75% | 33.75% | 33.75% | 16.25 pp | -26 | 4 | -6.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 105 | 48 | 57 | 45.71% | 45.71% | 45.71% | 4.29 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 105 | 46 | 59 | 43.81% | 43.81% | 43.81% | 6.19 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 105 | 43 | 62 | 40.95% | 40.95% | 40.95% | 9.05 pp | -19 | 5 | -3.80 |
| BTC Daily | xgb | XGBoost | 115 | 41 | 74 | 35.65% | 35.65% | 35.65% | 14.35 pp | -33 | 6 | -5.50 |
| BTC Daily | lstm | LSTM | 105 | 38 | 67 | 36.19% | 36.19% | 36.19% | 13.81 pp | -29 | 5 | -5.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 103 | 56 | 47 | 54.37% | 54.37% | 54.37% | 4.37 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 103 | 50 | 53 | 48.54% | 48.54% | 48.54% | 1.46 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 103 | 49 | 54 | 47.57% | 47.57% | 47.57% | 2.43 pp | -5 | 8 | -0.62 |
| BTC Market Hours | lstm | LSTM | 103 | 42 | 61 | 40.78% | 40.78% | 40.78% | 9.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | transformer | Transformer | 103 | 41 | 62 | 39.81% | 39.81% | 39.81% | 10.19 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 103 | 39 | 64 | 37.86% | 37.86% | 37.86% | 12.14 pp | -25 | 8 | -3.12 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 103 | 51 | 52 | 49.51% | 49.51% | 49.51% | 0.49 pp | -1 | 9 | -0.11 |
| BTC Market Hours Daily | rf | RandomForest | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | transformer | Transformer | 103 | 46 | 57 | 44.66% | 44.66% | 44.66% | 5.34 pp | -11 | 9 | -1.22 |
| BTC Market Hours Daily | nn | NN | 103 | 44 | 59 | 42.72% | 42.72% | 42.72% | 7.28 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 103 | 40 | 63 | 38.83% | 38.83% | 38.83% | 11.17 pp | -23 | 9 | -2.56 |
| BTC Market Hours Daily | xgb | XGBoost | 103 | 38 | 65 | 36.89% | 36.89% | 36.89% | 13.11 pp | -27 | 9 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

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
