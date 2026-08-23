# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T16:24:12.104010+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 103 | 2 | 101 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 98 | 38 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 64 | 26 | 38 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 64 | 26 | 38 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 16 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 16 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 0 | 16 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 11:00:00+00:00 | 16 | 0 | 16 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| BTC Market Hours | rf | RandomForest | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| BTC Market Hours | xgb | XGBoost | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| BTC Market Hours | nn | NN | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 2 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Daily | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Market Hours Daily | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours Daily | nn | NN | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Hourly | lstm | LSTM | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | transformer | Transformer | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | xgb | XGBoost | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 28 | 11 | 17 | 39.29% | 39.29% | 39.29% | 10.71 pp | -6 | 2 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 3 | -3.33 |
| BTC Daily | xgb | XGBoost | 38 | 13 | 25 | 34.21% | 34.21% | 34.21% | 15.79 pp | -12 | 3 | -4.00 |
| Consolidated Hourly | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |
| BTC Daily | lstm | LSTM | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 2 | -6.00 |
| BTC Market Hours | lstm | LSTM | 26 | 6 | 20 | 23.08% | 23.08% | 23.08% | 26.92 pp | -14 | 2 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | transformer | Transformer | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | xgb | XGBoost | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Daily | nn | NN | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Daily | transformer | Transformer | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 2 | 1.00 |
| BTC Daily | rf | RandomForest | 28 | 11 | 17 | 39.29% | 39.29% | 39.29% | 10.71 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 38 | 13 | 25 | 34.21% | 34.21% | 34.21% | 15.79 pp | -12 | 3 | -4.00 |
| BTC Daily | lstm | LSTM | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 2 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| BTC Market Hours | xgb | XGBoost | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 2 | 2.00 |
| BTC Market Hours | nn | NN | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 2 | 1.00 |
| BTC Market Hours | transformer | Transformer | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 2 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 2 | -2.00 |
| BTC Market Hours | lstm | LSTM | 26 | 6 | 20 | 23.08% | 23.08% | 23.08% | 26.92 pp | -14 | 2 | -7.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 26 | 15 | 11 | 57.69% | 57.69% | 57.69% | 7.69 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | xgb | XGBoost | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 26 | 13 | 13 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | nn | NN | 26 | 12 | 14 | 46.15% | 46.15% | 46.15% | 3.85 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | lstm | LSTM | 26 | 8 | 18 | 30.77% | 30.77% | 30.77% | 19.23 pp | -10 | 3 | -3.33 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Hourly | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 16 | 10 | 6 | 62.50% | 62.50% | 62.50% | 12.50 pp | 4 | 2 | 2.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 16 | 9 | 7 | 56.25% | 56.25% | 56.25% | 6.25 pp | 2 | 2 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 16 | 8 | 8 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 16 | 7 | 9 | 43.75% | 43.75% | 43.75% | 6.25 pp | -2 | 2 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 16 | 4 | 12 | 25.00% | 25.00% | 25.00% | 25.00 pp | -8 | 2 | -4.00 |

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
