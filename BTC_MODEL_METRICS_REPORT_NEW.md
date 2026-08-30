# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T20:02:45.403096+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 134 | 74 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 170 | 110 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 19:00:00+00:00 | 197 | 98 | 99 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 19:00:00+00:00 | 196 | 97 | 99 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 77 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 77 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 0 | 77 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 06:00:00+00:00 | 77 | 0 | 77 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 98 | 54 | 44 | 55.10% | 55.10% | 55.10% | 5.10 pp | 10 | 8 | 1.25 |
| BTC Hourly | transformer | Transformer | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| BTC Hourly | nn | NN | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | rf | RandomForest | 98 | 48 | 50 | 48.98% | 48.98% | 48.98% | 1.02 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 8 | -0.50 |
| Consolidated Hourly | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | nn | NN | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 9 | -1.67 |
| BTC Daily | nn | NN | 100 | 45 | 55 | 45.00% | 45.00% | 45.00% | 5.00 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 100 | 45 | 55 | 45.00% | 45.00% | 45.00% | 5.00 pp | -10 | 5 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 98 | 41 | 57 | 41.84% | 41.84% | 41.84% | 8.16 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 9 | -2.11 |
| Consolidated Hourly | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 9 | -2.78 |
| BTC Market Hours | xgb | XGBoost | 98 | 37 | 61 | 37.76% | 37.76% | 37.76% | 12.24 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 74 | 30 | 44 | 40.54% | 40.54% | 40.54% | 9.46 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 100 | 39 | 61 | 39.00% | 39.00% | 39.00% | 11.00 pp | -22 | 5 | -4.40 |
| BTC Hourly | lstm | LSTM | 74 | 26 | 48 | 35.14% | 35.14% | 35.14% | 14.86 pp | -22 | 4 | -5.50 |
| BTC Daily | lstm | LSTM | 100 | 36 | 64 | 36.00% | 36.00% | 36.00% | 14.00 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 110 | 38 | 72 | 34.55% | 34.55% | 34.55% | 15.45 pp | -34 | 6 | -5.67 |
| BTC Hourly | xgb | XGBoost | 74 | 25 | 49 | 33.78% | 33.78% | 33.78% | 16.22 pp | -24 | 4 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 74 | 39 | 35 | 52.70% | 52.70% | 52.70% | 2.70 pp | 4 | 4 | 1.00 |
| BTC Hourly | nn | NN | 74 | 37 | 37 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 74 | 33 | 41 | 44.59% | 44.59% | 44.59% | 5.41 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 74 | 30 | 44 | 40.54% | 40.54% | 40.54% | 9.46 pp | -14 | 4 | -3.50 |
| BTC Hourly | lstm | LSTM | 74 | 26 | 48 | 35.14% | 35.14% | 35.14% | 14.86 pp | -22 | 4 | -5.50 |
| BTC Hourly | xgb | XGBoost | 74 | 25 | 49 | 33.78% | 33.78% | 33.78% | 16.22 pp | -24 | 4 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 100 | 47 | 53 | 47.00% | 47.00% | 47.00% | 3.00 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 100 | 45 | 55 | 45.00% | 45.00% | 45.00% | 5.00 pp | -10 | 5 | -2.00 |
| BTC Daily | transformer | Transformer | 100 | 45 | 55 | 45.00% | 45.00% | 45.00% | 5.00 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 100 | 39 | 61 | 39.00% | 39.00% | 39.00% | 11.00 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 100 | 36 | 64 | 36.00% | 36.00% | 36.00% | 14.00 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 110 | 38 | 72 | 34.55% | 34.55% | 34.55% | 15.45 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 98 | 54 | 44 | 55.10% | 55.10% | 55.10% | 5.10 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 98 | 48 | 50 | 48.98% | 48.98% | 48.98% | 1.02 pp | -2 | 8 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 98 | 47 | 51 | 47.96% | 47.96% | 47.96% | 2.04 pp | -4 | 8 | -0.50 |
| BTC Market Hours | lstm | LSTM | 98 | 41 | 57 | 41.84% | 41.84% | 41.84% | 8.16 pp | -16 | 8 | -2.00 |
| BTC Market Hours | transformer | Transformer | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 8 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 98 | 37 | 61 | 37.76% | 37.76% | 37.76% | 12.24 pp | -24 | 8 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | rf | RandomForest | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 9 | -2.78 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 77 | 42 | 35 | 54.55% | 54.55% | 54.55% | 4.55 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 77 | 39 | 38 | 50.65% | 50.65% | 50.65% | 0.65 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 77 | 35 | 42 | 45.45% | 45.45% | 45.45% | 4.55 pp | -7 | 8 | -0.88 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 77 | 30 | 47 | 38.96% | 38.96% | 38.96% | 11.04 pp | -17 | 8 | -2.12 |

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
