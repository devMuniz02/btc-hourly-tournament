# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T15:11:41.348797+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 14 | 88 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 110 | 50 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 14:00:00+00:00 | 93 | 38 | 55 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 14:00:00+00:00 | 93 | 38 | 55 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 27 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 11:00:00+00:00 | 27 | 0 | 27 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| BTC Market Hours | nn | NN | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | nn | NN | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | rf | RandomForest | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | lstm | LSTM | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | transformer | Transformer | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 2 | -2.00 |
| BTC Hourly | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 38 | 11 | 27 | 28.95% | 28.95% | 28.95% | 21.05 pp | -16 | 4 | -4.00 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |
| BTC Daily | xgb | XGBoost | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 3 | -4.67 |
| BTC Market Hours | lstm | LSTM | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 2 | -6.00 |
| BTC Hourly | xgb | XGBoost | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 14 | 7 | 7 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 14 | 5 | 9 | 35.71% | 35.71% | 35.71% | 14.29 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 14 | 4 | 10 | 28.57% | 28.57% | 28.57% | 21.43 pp | -6 | 1 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 2 | 3.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | nn | NN | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | rf | RandomForest | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 2 | -2.00 |
| BTC Daily | xgb | XGBoost | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 3 | -4.67 |
| BTC Daily | lstm | LSTM | 40 | 14 | 26 | 35.00% | 35.00% | 35.00% | 15.00 pp | -12 | 2 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 3 | 0.67 |
| BTC Market Hours | rf | RandomForest | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | transformer | Transformer | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 3 | -0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours | xgb | XGBoost | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 3 | -4.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 38 | 19 | 19 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | nn | NN | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 38 | 17 | 21 | 44.74% | 44.74% | 44.74% | 5.26 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 38 | 11 | 27 | 28.95% | 28.95% | 28.95% | 21.05 pp | -16 | 4 | -4.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Hourly | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 27 | 17 | 10 | 62.96% | 62.96% | 62.96% | 12.96 pp | 7 | 3 | 2.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 27 | 15 | 12 | 55.56% | 55.56% | 55.56% | 5.56 pp | 3 | 3 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 27 | 14 | 13 | 51.85% | 51.85% | 51.85% | 1.85 pp | 1 | 3 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 27 | 11 | 16 | 40.74% | 40.74% | 40.74% | 9.26 pp | -5 | 3 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 27 | 7 | 20 | 25.93% | 25.93% | 25.93% | 24.07 pp | -13 | 3 | -4.33 |

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
