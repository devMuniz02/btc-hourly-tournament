# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-23T01:59:01.625134+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 88 | 2 | 86 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 74 | 29 | 45 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 51 | 17 | 34 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 00:00:00+00:00 | 51 | 17 | 34 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 9 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 0 | 9 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-19 04:00:00+00:00 | 9 | 0 | 9 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| BTC Market Hours | xgb | XGBoost | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| BTC Daily | nn | NN | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 2 | 1.50 |
| BTC Market Hours Daily | rf | RandomForest | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| BTC Market Hours | nn | NN | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | nn | NN | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Daily | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| BTC Daily | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 3 | -1.67 |
| BTC Hourly | lstm | LSTM | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | nn | NN | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | transformer | Transformer | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | xgb | XGBoost | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |
| BTC Market Hours | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |
| BTC Daily | lstm | LSTM | 19 | 5 | 14 | 26.32% | 26.32% | 26.32% | 23.68 pp | -9 | 2 | -4.50 |

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
| BTC Daily | nn | NN | 19 | 11 | 8 | 57.89% | 57.89% | 57.89% | 7.89 pp | 3 | 2 | 1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Daily | transformer | Transformer | 19 | 10 | 9 | 52.63% | 52.63% | 52.63% | 2.63 pp | 1 | 2 | 0.50 |
| BTC Daily | rf | RandomForest | 19 | 9 | 10 | 47.37% | 47.37% | 47.37% | 2.63 pp | -1 | 2 | -0.50 |
| BTC Daily | xgb | XGBoost | 29 | 12 | 17 | 41.38% | 41.38% | 41.38% | 8.62 pp | -5 | 3 | -1.67 |
| BTC Daily | lstm | LSTM | 19 | 5 | 14 | 26.32% | 26.32% | 26.32% | 23.68 pp | -9 | 2 | -4.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| BTC Market Hours | xgb | XGBoost | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 2 | 2.50 |
| BTC Market Hours | nn | NN | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 2 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | transformer | Transformer | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 2 | -0.50 |
| BTC Market Hours | lstm | LSTM | 17 | 5 | 12 | 29.41% | 29.41% | 29.41% | 20.59 pp | -7 | 2 | -3.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 17 | 11 | 6 | 64.71% | 64.71% | 64.71% | 14.71 pp | 5 | 3 | 1.67 |
| BTC Market Hours Daily | rf | RandomForest | 17 | 10 | 7 | 58.82% | 58.82% | 58.82% | 8.82 pp | 3 | 3 | 1.00 |
| BTC Market Hours Daily | nn | NN | 17 | 9 | 8 | 52.94% | 52.94% | 52.94% | 2.94 pp | 1 | 3 | 0.33 |
| BTC Market Hours Daily | transformer | Transformer | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | xgb | XGBoost | 17 | 8 | 9 | 47.06% | 47.06% | 47.06% | 2.94 pp | -1 | 3 | -0.33 |
| BTC Market Hours Daily | lstm | LSTM | 17 | 6 | 11 | 35.29% | 35.29% | 35.29% | 14.71 pp | -5 | 3 | -1.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Hourly | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Hourly | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 2 | 1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 2 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 2 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 2 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 2 | -2.50 |

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
