# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T09:46:25.245961+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 126 | 66 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 161 | 101 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 180 | 89 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 180 | 89 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 89 | 50 | 39 | 56.18% | 56.18% | 56.18% | 6.18 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | rf | RandomForest | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 7 | 0.14 |
| BTC Hourly | nn | NN | 66 | 33 | 33 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 89 | 44 | 45 | 49.44% | 49.44% | 49.44% | 0.56 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 8 | -0.88 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 8 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| BTC Daily | nn | NN | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 89 | 37 | 52 | 41.57% | 41.57% | 41.57% | 8.43 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Market Hours Daily | xgb | XGBoost | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 8 | -2.88 |
| BTC Market Hours | xgb | XGBoost | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 91 | 35 | 56 | 38.46% | 38.46% | 38.46% | 11.54 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 5 | -5.00 |
| BTC Hourly | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 101 | 32 | 69 | 31.68% | 31.68% | 31.68% | 18.32 pp | -37 | 6 | -6.17 |
| BTC Hourly | xgb | XGBoost | 66 | 21 | 45 | 31.82% | 31.82% | 31.82% | 18.18 pp | -24 | 3 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Hourly | nn | NN | 66 | 33 | 33 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 66 | 27 | 39 | 40.91% | 40.91% | 40.91% | 9.09 pp | -12 | 3 | -4.00 |
| BTC Hourly | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 66 | 21 | 45 | 31.82% | 31.82% | 31.82% | 18.18 pp | -24 | 3 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| BTC Daily | nn | NN | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 91 | 35 | 56 | 38.46% | 38.46% | 38.46% | 11.54 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 91 | 33 | 58 | 36.26% | 36.26% | 36.26% | 13.74 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 101 | 32 | 69 | 31.68% | 31.68% | 31.68% | 18.32 pp | -37 | 6 | -6.17 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 89 | 50 | 39 | 56.18% | 56.18% | 56.18% | 6.18 pp | 11 | 7 | 1.57 |
| BTC Market Hours | rf | RandomForest | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 89 | 44 | 45 | 49.44% | 49.44% | 49.44% | 0.56 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 89 | 37 | 52 | 41.57% | 41.57% | 41.57% | 8.43 pp | -15 | 7 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | nn | NN | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 8 | -2.88 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

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
