# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T06:46:09.171883+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 8 | 94 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 104 | 44 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 84 | 32 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 84 | 32 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T06:00:00+00:00 | 22 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T06:00:00+00:00 | 22 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T06:00:00+00:00 | 22 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-20T06:00:00+00:00 | 23 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 34 | 20 | 14 | 58.82% | 58.82% | 58.82% | 8.82 pp | 6 | 2 | 3.00 |
| Consolidated Hourly | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| BTC Market Hours | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| BTC Daily | nn | NN | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 2 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours | nn | NN | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 34 | 17 | 17 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Market Hours | transformer | Transformer | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | xgb | XGBoost | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| Consolidated Hourly | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| BTC Daily | rf | RandomForest | 34 | 14 | 20 | 41.18% | 41.18% | 41.18% | 8.82 pp | -6 | 2 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 3 | -3.33 |
| BTC Market Hours | lstm | LSTM | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 2 | -5.00 |
| BTC Hourly | rf | RandomForest | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | transformer | Transformer | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 8 | 4 | 4 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 8 | 3 | 5 | 37.50% | 37.50% | 37.50% | 12.50 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | transformer | Transformer | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |
| BTC Hourly | xgb | XGBoost | 8 | 1 | 7 | 12.50% | 12.50% | 12.50% | 37.50 pp | -6 | 1 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 34 | 20 | 14 | 58.82% | 58.82% | 58.82% | 8.82 pp | 6 | 2 | 3.00 |
| BTC Daily | nn | NN | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 2 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 34 | 17 | 17 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 2 | 0.00 |
| BTC Daily | rf | RandomForest | 34 | 14 | 20 | 41.18% | 41.18% | 41.18% | 8.82 pp | -6 | 2 | -3.00 |
| BTC Daily | xgb | XGBoost | 44 | 17 | 27 | 38.64% | 38.64% | 38.64% | 11.36 pp | -10 | 3 | -3.33 |
| BTC Daily | lstm | LSTM | 34 | 12 | 22 | 35.29% | 35.29% | 35.29% | 14.71 pp | -10 | 2 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 3 | 1.33 |
| BTC Market Hours | nn | NN | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 3 | 0.67 |
| BTC Market Hours | transformer | Transformer | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | xgb | XGBoost | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 32 | 15 | 17 | 46.88% | 46.88% | 46.88% | 3.12 pp | -2 | 3 | -0.67 |
| BTC Market Hours | lstm | LSTM | 32 | 11 | 21 | 34.38% | 34.38% | 34.38% | 15.62 pp | -10 | 3 | -3.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| BTC Market Hours Daily | transformer | Transformer | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | nn | NN | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
