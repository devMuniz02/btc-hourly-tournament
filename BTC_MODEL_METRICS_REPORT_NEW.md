# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T12:22:19.897455+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 128 | 68 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 164 | 104 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 183 | 92 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 182 | 91 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 92 | 51 | 41 | 55.43% | 55.43% | 55.43% | 5.43 pp | 10 | 8 | 1.25 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| BTC Hourly | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 8 | 0.38 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | rf | RandomForest | 92 | 46 | 46 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | nn | NN | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 8 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | rf | RandomForest | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 8 | -0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Market Hours Daily | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Market Hours | lstm | LSTM | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 3 | -2.00 |
| BTC Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 5 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 8 | -2.12 |
| BTC Market Hours | transformer | Transformer | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 91 | 34 | 57 | 37.36% | 37.36% | 37.36% | 12.64 pp | -23 | 8 | -2.88 |
| BTC Hourly | rf | RandomForest | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 94 | 34 | 60 | 36.17% | 36.17% | 36.17% | 13.83 pp | -26 | 5 | -5.20 |
| BTC Hourly | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 104 | 34 | 70 | 32.69% | 32.69% | 32.69% | 17.31 pp | -36 | 6 | -6.00 |
| BTC Hourly | xgb | XGBoost | 68 | 21 | 47 | 30.88% | 30.88% | 30.88% | 19.12 pp | -26 | 3 | -8.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 3 | 0.67 |
| BTC Hourly | nn | NN | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 3 | -2.00 |
| BTC Hourly | rf | RandomForest | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 3 | -4.00 |
| BTC Hourly | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 68 | 21 | 47 | 30.88% | 30.88% | 30.88% | 19.12 pp | -26 | 3 | -8.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 94 | 34 | 60 | 36.17% | 36.17% | 36.17% | 13.83 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 104 | 34 | 70 | 32.69% | 32.69% | 32.69% | 17.31 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 92 | 51 | 41 | 55.43% | 55.43% | 55.43% | 5.43 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 92 | 46 | 46 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 8 | -0.25 |
| BTC Market Hours | lstm | LSTM | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours | transformer | Transformer | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | rf | RandomForest | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 91 | 34 | 57 | 37.36% | 37.36% | 37.36% | 12.64 pp | -23 | 8 | -2.88 |

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
