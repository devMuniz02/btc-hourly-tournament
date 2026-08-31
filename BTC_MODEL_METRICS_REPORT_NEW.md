# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T02:07:03.337261+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 139 | 79 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 174 | 114 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 206 | 102 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 206 | 102 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 82 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 11:00:00+00:00 | 82 | 0 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 102 | 56 | 46 | 54.90% | 54.90% | 54.90% | 4.90 pp | 10 | 8 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| BTC Hourly | transformer | Transformer | 79 | 41 | 38 | 51.90% | 51.90% | 51.90% | 1.90 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 102 | 51 | 51 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | nn | NN | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 4 | -0.25 |
| BTC Market Hours | rf | RandomForest | 102 | 49 | 53 | 48.04% | 48.04% | 48.04% | 1.96 pp | -4 | 8 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 104 | 50 | 54 | 48.08% | 48.08% | 48.08% | 1.92 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | transformer | Transformer | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | nn | NN | 102 | 44 | 58 | 43.14% | 43.14% | 43.14% | 6.86 pp | -14 | 9 | -1.56 |
| BTC Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 4 | -2.25 |
| BTC Market Hours | lstm | LSTM | 102 | 42 | 60 | 41.18% | 41.18% | 41.18% | 8.82 pp | -18 | 8 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 9 | -2.44 |
| BTC Market Hours | transformer | Transformer | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 8 | -2.75 |
| BTC Daily | transformer | Transformer | 104 | 45 | 59 | 43.27% | 43.27% | 43.27% | 6.73 pp | -14 | 5 | -2.80 |
| BTC Market Hours | xgb | XGBoost | 102 | 39 | 63 | 38.24% | 38.24% | 38.24% | 11.76 pp | -24 | 8 | -3.00 |
| BTC Market Hours Daily | xgb | XGBoost | 102 | 37 | 65 | 36.27% | 36.27% | 36.27% | 13.73 pp | -28 | 9 | -3.11 |
| BTC Hourly | rf | RandomForest | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 104 | 42 | 62 | 40.38% | 40.38% | 40.38% | 9.62 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 104 | 38 | 66 | 36.54% | 36.54% | 36.54% | 13.46 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 114 | 40 | 74 | 35.09% | 35.09% | 35.09% | 14.91 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 79 | 41 | 38 | 51.90% | 51.90% | 51.90% | 1.90 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 79 | 39 | 40 | 49.37% | 49.37% | 49.37% | 0.63 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 4 | -2.25 |
| BTC Hourly | rf | RandomForest | 79 | 33 | 46 | 41.77% | 41.77% | 41.77% | 8.23 pp | -13 | 4 | -3.25 |
| BTC Hourly | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 104 | 50 | 54 | 48.08% | 48.08% | 48.08% | 1.92 pp | -4 | 5 | -0.80 |
| BTC Daily | nn | NN | 104 | 48 | 56 | 46.15% | 46.15% | 46.15% | 3.85 pp | -8 | 5 | -1.60 |
| BTC Daily | transformer | Transformer | 104 | 45 | 59 | 43.27% | 43.27% | 43.27% | 6.73 pp | -14 | 5 | -2.80 |
| BTC Daily | rf | RandomForest | 104 | 42 | 62 | 40.38% | 40.38% | 40.38% | 9.62 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 104 | 38 | 66 | 36.54% | 36.54% | 36.54% | 13.46 pp | -28 | 5 | -5.60 |
| BTC Daily | xgb | XGBoost | 114 | 40 | 74 | 35.09% | 35.09% | 35.09% | 14.91 pp | -34 | 6 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 102 | 56 | 46 | 54.90% | 54.90% | 54.90% | 4.90 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 102 | 49 | 53 | 48.04% | 48.04% | 48.04% | 1.96 pp | -4 | 8 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 102 | 48 | 54 | 47.06% | 47.06% | 47.06% | 2.94 pp | -6 | 8 | -0.75 |
| BTC Market Hours | lstm | LSTM | 102 | 42 | 60 | 41.18% | 41.18% | 41.18% | 8.82 pp | -18 | 8 | -2.25 |
| BTC Market Hours | transformer | Transformer | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 8 | -2.75 |
| BTC Market Hours | xgb | XGBoost | 102 | 39 | 63 | 38.24% | 38.24% | 38.24% | 11.76 pp | -24 | 8 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 102 | 51 | 51 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | transformer | Transformer | 102 | 46 | 56 | 45.10% | 45.10% | 45.10% | 4.90 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | nn | NN | 102 | 44 | 58 | 43.14% | 43.14% | 43.14% | 6.86 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 102 | 40 | 62 | 39.22% | 39.22% | 39.22% | 10.78 pp | -22 | 9 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 102 | 37 | 65 | 36.27% | 36.27% | 36.27% | 13.73 pp | -28 | 9 | -3.11 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 82 | 45 | 37 | 54.88% | 54.88% | 54.88% | 4.88 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 82 | 33 | 49 | 40.24% | 40.24% | 40.24% | 9.76 pp | -16 | 8 | -2.00 |

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
