# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T08:56:46.823315+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 125 | 65 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 161 | 101 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 180 | 89 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 180 | 89 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 70 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 10:00:00+00:00 | 70 | 0 | 70 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 89 | 50 | 39 | 56.18% | 56.18% | 56.18% | 6.18 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 89 | 46 | 43 | 51.69% | 51.69% | 51.69% | 1.69 pp | 3 | 8 | 0.38 |
| BTC Hourly | nn | NN | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| BTC Market Hours | rf | RandomForest | 89 | 45 | 44 | 50.56% | 50.56% | 50.56% | 0.56 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 89 | 44 | 45 | 49.44% | 49.44% | 49.44% | 0.56 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 89 | 43 | 46 | 48.31% | 48.31% | 48.31% | 1.69 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 8 | -0.88 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 3 | -1.00 |
| BTC Daily | nn | NN | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 8 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 89 | 39 | 50 | 43.82% | 43.82% | 43.82% | 6.18 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 89 | 36 | 53 | 40.45% | 40.45% | 40.45% | 9.55 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 89 | 37 | 52 | 41.57% | 41.57% | 41.57% | 8.43 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |
| BTC Market Hours Daily | xgb | XGBoost | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 8 | -2.88 |
| BTC Market Hours | xgb | XGBoost | 89 | 34 | 55 | 38.20% | 38.20% | 38.20% | 11.80 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 91 | 35 | 56 | 38.46% | 38.46% | 38.46% | 11.54 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 91 | 32 | 59 | 35.16% | 35.16% | 35.16% | 14.84 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 101 | 32 | 69 | 31.68% | 31.68% | 31.68% | 18.32 pp | -37 | 6 | -6.17 |
| BTC Hourly | lstm | LSTM | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 65 | 21 | 44 | 32.31% | 32.31% | 32.31% | 17.69 pp | -23 | 3 | -7.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 65 | 31 | 34 | 47.69% | 47.69% | 47.69% | 2.31 pp | -3 | 3 | -1.00 |
| BTC Hourly | rf | RandomForest | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 65 | 21 | 44 | 32.31% | 32.31% | 32.31% | 17.69 pp | -23 | 3 | -7.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 5 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 91 | 41 | 50 | 45.05% | 45.05% | 45.05% | 4.95 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 91 | 35 | 56 | 38.46% | 38.46% | 38.46% | 11.54 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 91 | 32 | 59 | 35.16% | 35.16% | 35.16% | 14.84 pp | -27 | 5 | -5.40 |
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
| Consolidated Hourly | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Hourly | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 70 | 39 | 31 | 55.71% | 55.71% | 55.71% | 5.71 pp | 8 | 7 | 1.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 70 | 38 | 32 | 54.29% | 54.29% | 54.29% | 4.29 pp | 6 | 7 | 0.86 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 70 | 36 | 34 | 51.43% | 51.43% | 51.43% | 1.43 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 70 | 31 | 39 | 44.29% | 44.29% | 44.29% | 5.71 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 70 | 30 | 40 | 42.86% | 42.86% | 42.86% | 7.14 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 70 | 26 | 44 | 37.14% | 37.14% | 37.14% | 12.86 pp | -18 | 7 | -2.57 |

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
