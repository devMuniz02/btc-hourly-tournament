# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T17:49:40.896609+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 132 | 72 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 168 | 108 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 192 | 96 | 96 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 192 | 96 | 96 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 76 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T05:00:00+00:00 | 77 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 3 | 1.33 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| BTC Market Hours | nn | NN | 96 | 52 | 44 | 54.17% | 54.17% | 54.17% | 4.17 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 49 | 47 | 51.04% | 51.04% | 51.04% | 1.04 pp | 2 | 9 | 0.22 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | rf | RandomForest | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 8 | -0.50 |
| BTC Hourly | nn | NN | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 9 | -1.56 |
| BTC Daily | nn | NN | 98 | 45 | 53 | 45.92% | 45.92% | 45.92% | 4.08 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 8 | -1.75 |
| BTC Daily | transformer | Transformer | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 9 | -2.22 |
| BTC Market Hours | transformer | Transformer | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 8 | -2.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 3 | -2.67 |
| BTC Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 9 | -2.67 |
| BTC Market Hours | xgb | XGBoost | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 5 | -5.20 |
| BTC Hourly | rf | RandomForest | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 108 | 38 | 70 | 35.19% | 35.19% | 35.19% | 14.81 pp | -32 | 6 | -5.33 |
| BTC Hourly | lstm | LSTM | 72 | 26 | 46 | 36.11% | 36.11% | 36.11% | 13.89 pp | -20 | 3 | -6.67 |
| BTC Hourly | xgb | XGBoost | 72 | 23 | 49 | 31.94% | 31.94% | 31.94% | 18.06 pp | -26 | 3 | -8.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 3 | 1.33 |
| BTC Hourly | nn | NN | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 3 | -0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 3 | -2.67 |
| BTC Hourly | rf | RandomForest | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 3 | -5.33 |
| BTC Hourly | lstm | LSTM | 72 | 26 | 46 | 36.11% | 36.11% | 36.11% | 13.89 pp | -20 | 3 | -6.67 |
| BTC Hourly | xgb | XGBoost | 72 | 23 | 49 | 31.94% | 31.94% | 31.94% | 18.06 pp | -26 | 3 | -8.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 98 | 45 | 53 | 45.92% | 45.92% | 45.92% | 4.08 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 108 | 38 | 70 | 35.19% | 35.19% | 35.19% | 14.81 pp | -32 | 6 | -5.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 96 | 52 | 44 | 54.17% | 54.17% | 54.17% | 4.17 pp | 8 | 8 | 1.00 |
| BTC Market Hours | rf | RandomForest | 96 | 47 | 49 | 48.96% | 48.96% | 48.96% | 1.04 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 96 | 46 | 50 | 47.92% | 47.92% | 47.92% | 2.08 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 8 | -1.75 |
| BTC Market Hours | transformer | Transformer | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 96 | 37 | 59 | 38.54% | 38.54% | 38.54% | 11.46 pp | -22 | 8 | -2.75 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 96 | 49 | 47 | 51.04% | 51.04% | 51.04% | 1.04 pp | 2 | 9 | 0.22 |
| BTC Market Hours Daily | rf | RandomForest | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | transformer | Transformer | 96 | 44 | 52 | 45.83% | 45.83% | 45.83% | 4.17 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | nn | NN | 96 | 41 | 55 | 42.71% | 42.71% | 42.71% | 7.29 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 96 | 38 | 58 | 39.58% | 39.58% | 39.58% | 10.42 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 96 | 36 | 60 | 37.50% | 37.50% | 37.50% | 12.50 pp | -24 | 9 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 76 | 42 | 34 | 55.26% | 55.26% | 55.26% | 5.26 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 76 | 41 | 35 | 53.95% | 53.95% | 53.95% | 3.95 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 76 | 38 | 38 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
