# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T10:44:20.983334+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 162 | 102 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 181 | 90 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 181 | 90 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 90 | 51 | 39 | 56.67% | 56.67% | 56.67% | 6.67 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| BTC Hourly | nn | NN | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 47 | 43 | 52.22% | 52.22% | 52.22% | 2.22 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | rf | RandomForest | 90 | 45 | 45 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 90 | 44 | 46 | 48.89% | 48.89% | 48.89% | 1.11 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 90 | 43 | 47 | 47.78% | 47.78% | 47.78% | 2.22 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 8 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 92 | 42 | 50 | 45.65% | 45.65% | 45.65% | 4.35 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 8 | -2.00 |
| BTC Market Hours | transformer | Transformer | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 90 | 35 | 55 | 38.89% | 38.89% | 38.89% | 11.11 pp | -20 | 7 | -2.86 |
| BTC Market Hours Daily | xgb | XGBoost | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 3 | -3.33 |
| BTC Daily | rf | RandomForest | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 5 | -5.20 |
| BTC Hourly | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 102 | 33 | 69 | 32.35% | 32.35% | 32.35% | 17.65 pp | -36 | 6 | -6.00 |
| BTC Hourly | xgb | XGBoost | 66 | 21 | 45 | 31.82% | 31.82% | 31.82% | 18.18 pp | -24 | 3 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 66 | 34 | 32 | 51.52% | 51.52% | 51.52% | 1.52 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 66 | 31 | 35 | 46.97% | 46.97% | 46.97% | 3.03 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 66 | 28 | 38 | 42.42% | 42.42% | 42.42% | 7.58 pp | -10 | 3 | -3.33 |
| BTC Hourly | lstm | LSTM | 66 | 24 | 42 | 36.36% | 36.36% | 36.36% | 13.64 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 66 | 21 | 45 | 31.82% | 31.82% | 31.82% | 18.18 pp | -24 | 3 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 92 | 42 | 50 | 45.65% | 45.65% | 45.65% | 4.35 pp | -8 | 5 | -1.60 |
| BTC Daily | rf | RandomForest | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 5 | -4.00 |
| BTC Daily | lstm | LSTM | 92 | 33 | 59 | 35.87% | 35.87% | 35.87% | 14.13 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 102 | 33 | 69 | 32.35% | 32.35% | 32.35% | 17.65 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 90 | 51 | 39 | 56.67% | 56.67% | 56.67% | 6.67 pp | 12 | 7 | 1.71 |
| BTC Market Hours | rf | RandomForest | 90 | 45 | 45 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 90 | 44 | 46 | 48.89% | 48.89% | 48.89% | 1.11 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 7 | -1.71 |
| BTC Market Hours | transformer | Transformer | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 7 | -2.29 |
| BTC Market Hours | xgb | XGBoost | 90 | 35 | 55 | 38.89% | 38.89% | 38.89% | 11.11 pp | -20 | 7 | -2.86 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 90 | 47 | 43 | 52.22% | 52.22% | 52.22% | 2.22 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 90 | 43 | 47 | 47.78% | 47.78% | 47.78% | 2.22 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | nn | NN | 90 | 39 | 51 | 43.33% | 43.33% | 43.33% | 6.67 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 90 | 37 | 53 | 41.11% | 41.11% | 41.11% | 8.89 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 90 | 33 | 57 | 36.67% | 36.67% | 36.67% | 13.33 pp | -24 | 8 | -3.00 |

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
