# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T07:18:25.418151+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 124 | 64 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 159 | 99 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 178 | 87 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 178 | 87 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 87 | 49 | 38 | 56.32% | 56.32% | 56.32% | 6.32 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| BTC Hourly | nn | NN | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 8 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 8 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 8 | -1.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | nn | NN | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Market Hours | lstm | LSTM | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | lstm | LSTM | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | xgb | XGBoost | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 8 | -2.88 |
| BTC Market Hours | xgb | XGBoost | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |
| BTC Hourly | rf | RandomForest | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 5 | -4.60 |
| BTC Daily | lstm | LSTM | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 5 | -5.00 |
| BTC Hourly | lstm | LSTM | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 99 | 30 | 69 | 30.30% | 30.30% | 30.30% | 19.70 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 64 | 21 | 43 | 32.81% | 32.81% | 32.81% | 17.19 pp | -22 | 3 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| BTC Hourly | rf | RandomForest | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 3 | -4.00 |
| BTC Hourly | lstm | LSTM | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 64 | 21 | 43 | 32.81% | 32.81% | 32.81% | 17.19 pp | -22 | 3 | -7.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | nn | NN | 89 | 41 | 48 | 46.07% | 46.07% | 46.07% | 3.93 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 89 | 40 | 49 | 44.94% | 44.94% | 44.94% | 5.06 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 89 | 33 | 56 | 37.08% | 37.08% | 37.08% | 12.92 pp | -23 | 5 | -4.60 |
| BTC Daily | lstm | LSTM | 89 | 32 | 57 | 35.96% | 35.96% | 35.96% | 14.04 pp | -25 | 5 | -5.00 |
| BTC Daily | xgb | XGBoost | 99 | 30 | 69 | 30.30% | 30.30% | 30.30% | 19.70 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 87 | 49 | 38 | 56.32% | 56.32% | 56.32% | 6.32 pp | 11 | 7 | 1.57 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | rf | RandomForest | 87 | 43 | 44 | 49.43% | 49.43% | 49.43% | 0.57 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 7 | -1.57 |
| BTC Market Hours | transformer | Transformer | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 7 | -2.14 |
| BTC Market Hours | xgb | XGBoost | 87 | 33 | 54 | 37.93% | 37.93% | 37.93% | 12.07 pp | -21 | 7 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 8 | 0.62 |
| BTC Market Hours Daily | transformer | Transformer | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 8 | -1.12 |
| BTC Market Hours Daily | nn | NN | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | xgb | XGBoost | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 8 | -2.88 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |

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
