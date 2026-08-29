# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T04:26:48.677045+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 103 | 43 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 139 | 79 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 145 | 67 | 78 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 144 | 66 | 78 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 01:00:00+00:00 | 50 | 0 | 50 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | transformer | Transformer | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 6 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 2 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 6 | 0.33 |
| BTC Daily | transformer | Transformer | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 4 | 0.25 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Hourly | nn | NN | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 2 | -0.50 |
| BTC Market Hours | rf | RandomForest | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 6 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours | lstm | LSTM | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |
| BTC Hourly | lstm | LSTM | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 2 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 6 | -3.00 |
| BTC Daily | rf | RandomForest | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 69 | 24 | 45 | 34.78% | 34.78% | 34.78% | 15.22 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 79 | 26 | 53 | 32.91% | 32.91% | 32.91% | 17.09 pp | -27 | 5 | -5.40 |
| BTC Hourly | rf | RandomForest | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 43 | 14 | 29 | 32.56% | 32.56% | 32.56% | 17.44 pp | -15 | 2 | -7.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 43 | 22 | 21 | 51.16% | 51.16% | 51.16% | 1.16 pp | 1 | 2 | 0.50 |
| BTC Hourly | nn | NN | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 43 | 21 | 22 | 48.84% | 48.84% | 48.84% | 1.16 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 43 | 14 | 29 | 32.56% | 32.56% | 32.56% | 17.44 pp | -15 | 2 | -7.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 69 | 35 | 34 | 50.72% | 50.72% | 50.72% | 0.72 pp | 1 | 4 | 0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 69 | 33 | 36 | 47.83% | 47.83% | 47.83% | 2.17 pp | -3 | 4 | -0.75 |
| BTC Daily | rf | RandomForest | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 4 | -3.75 |
| BTC Daily | lstm | LSTM | 69 | 24 | 45 | 34.78% | 34.78% | 34.78% | 15.22 pp | -21 | 4 | -5.25 |
| BTC Daily | xgb | XGBoost | 79 | 26 | 53 | 32.91% | 32.91% | 32.91% | 17.09 pp | -27 | 5 | -5.40 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 6 | 1.50 |
| BTC Market Hours | rf | RandomForest | 67 | 32 | 35 | 47.76% | 47.76% | 47.76% | 2.24 pp | -3 | 6 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 6 | -1.50 |
| BTC Market Hours | lstm | LSTM | 67 | 27 | 40 | 40.30% | 40.30% | 40.30% | 9.70 pp | -13 | 6 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 66 | 35 | 31 | 53.03% | 53.03% | 53.03% | 3.03 pp | 4 | 6 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 6 | 0.33 |
| BTC Market Hours Daily | rf | RandomForest | 66 | 29 | 37 | 43.94% | 43.94% | 43.94% | 6.06 pp | -8 | 6 | -1.33 |
| BTC Market Hours Daily | nn | NN | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 6 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 6 | 0.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 18 | 32 | 36.00% | 36.00% | 36.00% | 14.00 pp | -14 | 6 | -2.33 |

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
