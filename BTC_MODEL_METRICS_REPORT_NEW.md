# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T17:12:11.230214+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 167 | 107 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 191 | 95 | 96 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 16:00:00+00:00 | 191 | 95 | 96 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 75 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 0 | 75 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 04:00:00+00:00 | 75 | 0 | 75 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 3 | 1.33 |
| BTC Market Hours | nn | NN | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 8 | 1.12 |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| BTC Market Hours | rf | RandomForest | 95 | 47 | 48 | 49.47% | 49.47% | 49.47% | 0.53 pp | -1 | 8 | -0.12 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 8 | -0.38 |
| BTC Hourly | nn | NN | 72 | 35 | 37 | 48.61% | 48.61% | 48.61% | 1.39 pp | -2 | 3 | -0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | nn | NN | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 9 | -1.67 |
| BTC Daily | nn | NN | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |
| BTC Daily | transformer | Transformer | 97 | 43 | 54 | 44.33% | 44.33% | 44.33% | 5.67 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 9 | -2.33 |
| BTC Market Hours | transformer | Transformer | 95 | 38 | 57 | 40.00% | 40.00% | 40.00% | 10.00 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 72 | 32 | 40 | 44.44% | 44.44% | 44.44% | 5.56 pp | -8 | 3 | -2.67 |
| BTC Market Hours Daily | xgb | XGBoost | 95 | 35 | 60 | 36.84% | 36.84% | 36.84% | 13.16 pp | -25 | 9 | -2.78 |
| BTC Daily | rf | RandomForest | 97 | 38 | 59 | 39.18% | 39.18% | 39.18% | 10.82 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 5 | -5.00 |
| BTC Hourly | rf | RandomForest | 72 | 28 | 44 | 38.89% | 38.89% | 38.89% | 11.11 pp | -16 | 3 | -5.33 |
| BTC Daily | xgb | XGBoost | 107 | 37 | 70 | 34.58% | 34.58% | 34.58% | 15.42 pp | -33 | 6 | -5.50 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 97 | 43 | 54 | 44.33% | 44.33% | 44.33% | 5.67 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 97 | 38 | 59 | 39.18% | 39.18% | 39.18% | 10.82 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 107 | 37 | 70 | 34.58% | 34.58% | 34.58% | 15.42 pp | -33 | 6 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 95 | 52 | 43 | 54.74% | 54.74% | 54.74% | 4.74 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 95 | 47 | 48 | 49.47% | 49.47% | 49.47% | 0.53 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 95 | 46 | 49 | 48.42% | 48.42% | 48.42% | 1.58 pp | -3 | 8 | -0.38 |
| BTC Market Hours | lstm | LSTM | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 95 | 38 | 57 | 40.00% | 40.00% | 40.00% | 10.00 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 48 | 47 | 50.53% | 50.53% | 50.53% | 0.53 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | rf | RandomForest | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 95 | 40 | 55 | 42.11% | 42.11% | 42.11% | 7.89 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 95 | 35 | 60 | 36.84% | 36.84% | 36.84% | 13.16 pp | -25 | 9 | -2.78 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Hourly | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 75 | 41 | 34 | 54.67% | 54.67% | 54.67% | 4.67 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 75 | 40 | 35 | 53.33% | 53.33% | 53.33% | 3.33 pp | 5 | 8 | 0.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 8 | -0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 8 | -1.38 |
| Consolidated Daily/Hourly Refresh | nn | NN | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 8 | -2.12 |

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
