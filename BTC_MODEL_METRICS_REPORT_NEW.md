# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T07:44:22.614664+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 9 | 93 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 105 | 45 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 85 | 33 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 84 | 32 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 22 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 22 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 0 | 22 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 0 | 22 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| BTC Daily | transformer | Transformer | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 2 | 2.50 |
| BTC Market Hours | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 3 | 1.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 32 | 18 | 14 | 56.25% | 56.25% | 56.25% | 6.25 pp | 4 | 4 | 1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 32 | 17 | 15 | 53.12% | 53.12% | 53.12% | 3.12 pp | 2 | 4 | 0.50 |
| BTC Daily | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 32 | 16 | 16 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | xgb | XGBoost | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 32 | 14 | 18 | 43.75% | 43.75% | 43.75% | 6.25 pp | -4 | 4 | -1.00 |
| BTC Daily | rf | RandomForest | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 2 | -2.50 |
| Consolidated Hourly | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| BTC Hourly | nn | NN | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 32 | 10 | 22 | 31.25% | 31.25% | 31.25% | 18.75 pp | -12 | 4 | -3.00 |
| BTC Daily | xgb | XGBoost | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 3 | -3.67 |
| BTC Market Hours | lstm | LSTM | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |
| BTC Daily | lstm | LSTM | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 9 | 1 | 8 | 11.11% | 11.11% | 11.11% | 38.89 pp | -7 | 1 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| BTC Hourly | lstm | LSTM | 9 | 4 | 5 | 44.44% | 44.44% | 44.44% | 5.56 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 9 | 2 | 7 | 22.22% | 22.22% | 22.22% | 27.78 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 9 | 1 | 8 | 11.11% | 11.11% | 11.11% | 38.89 pp | -7 | 1 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 2 | 2.50 |
| BTC Daily | nn | NN | 35 | 18 | 17 | 51.43% | 51.43% | 51.43% | 1.43 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 35 | 17 | 18 | 48.57% | 48.57% | 48.57% | 1.43 pp | -1 | 2 | -0.50 |
| BTC Daily | rf | RandomForest | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 3 | -3.67 |
| BTC Daily | lstm | LSTM | 35 | 12 | 23 | 34.29% | 34.29% | 34.29% | 15.71 pp | -11 | 2 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 3 | 1.67 |
| BTC Market Hours | nn | NN | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 3 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | xgb | XGBoost | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 3 | -0.33 |
| BTC Market Hours | lstm | LSTM | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 3 | -3.67 |

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
