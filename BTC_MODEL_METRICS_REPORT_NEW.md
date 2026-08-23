# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T06:52:09.658327+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 05:00:00+00:00 | 92 | 1 | 91 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 05:00:00+00:00 | 82 | 32 | 50 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 54 | 20 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 54 | 20 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 12 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 07:00:00+00:00 | 12 | 0 | 12 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 20 | 14 | 6 | 70.00% | 70.00% | 70.00% | 20.00 pp | 8 | 2 | 4.00 |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| BTC Market Hours | rf | RandomForest | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 2 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | rf | RandomForest | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 3 | -2.00 |
| BTC Daily | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |
| BTC Market Hours | lstm | LSTM | 20 | 5 | 15 | 25.00% | 25.00% | 25.00% | 25.00 pp | -10 | 2 | -5.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 2 | 1.00 |
| BTC Daily | transformer | Transformer | 22 | 10 | 12 | 45.45% | 45.45% | 45.45% | 4.55 pp | -2 | 2 | -1.00 |
| BTC Daily | rf | RandomForest | 22 | 8 | 14 | 36.36% | 36.36% | 36.36% | 13.64 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 22 | 6 | 16 | 27.27% | 27.27% | 27.27% | 22.73 pp | -10 | 2 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | xgb | XGBoost | 20 | 14 | 6 | 70.00% | 70.00% | 70.00% | 20.00 pp | 8 | 2 | 4.00 |
| BTC Market Hours | rf | RandomForest | 20 | 12 | 8 | 60.00% | 60.00% | 60.00% | 10.00 pp | 4 | 2 | 2.00 |
| BTC Market Hours | nn | NN | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 2 | 1.00 |
| BTC Market Hours | transformer | Transformer | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 2 | -2.00 |
| BTC Market Hours | lstm | LSTM | 20 | 5 | 15 | 25.00% | 25.00% | 25.00% | 25.00 pp | -10 | 2 | -5.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | rf | RandomForest | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 20 | 11 | 9 | 55.00% | 55.00% | 55.00% | 5.00 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | lstm | LSTM | 20 | 7 | 13 | 35.00% | 35.00% | 35.00% | 15.00 pp | -6 | 3 | -2.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 12 | 9 | 3 | 75.00% | 75.00% | 75.00% | 25.00 pp | 6 | 2 | 3.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 12 | 7 | 5 | 58.33% | 58.33% | 58.33% | 8.33 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 12 | 3 | 9 | 25.00% | 25.00% | 25.00% | 25.00 pp | -6 | 2 | -3.00 |

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
