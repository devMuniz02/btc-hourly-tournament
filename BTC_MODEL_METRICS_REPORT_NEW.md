# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T07:27:39.135749+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 160 | 100 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 179 | 88 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 178 | 87 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 69 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 09:00:00+00:00 | 69 | 0 | 69 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 88 | 50 | 38 | 56.82% | 56.82% | 56.82% | 6.82 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 69 | 39 | 30 | 56.52% | 56.52% | 56.52% | 6.52 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 69 | 37 | 32 | 53.62% | 53.62% | 53.62% | 3.62 pp | 5 | 7 | 0.71 |
| BTC Hourly | nn | NN | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 87 | 46 | 41 | 52.87% | 52.87% | 52.87% | 2.87 pp | 5 | 8 | 0.62 |
| Consolidated Hourly | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 7 | 0.43 |
| BTC Market Hours | rf | RandomForest | 88 | 44 | 44 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 88 | 43 | 45 | 48.86% | 48.86% | 48.86% | 1.14 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 87 | 42 | 45 | 48.28% | 48.28% | 48.28% | 1.72 pp | -3 | 8 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 7 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 87 | 39 | 48 | 44.83% | 44.83% | 44.83% | 5.17 pp | -9 | 8 | -1.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 87 | 38 | 49 | 43.68% | 43.68% | 43.68% | 6.32 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 7 | -1.57 |
| BTC Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 87 | 36 | 51 | 41.38% | 41.38% | 41.38% | 8.62 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 88 | 37 | 51 | 42.05% | 42.05% | 42.05% | 7.95 pp | -14 | 7 | -2.00 |
| Consolidated Hourly | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 69 | 25 | 44 | 36.23% | 36.23% | 36.23% | 13.77 pp | -19 | 7 | -2.71 |
| BTC Market Hours | xgb | XGBoost | 88 | 34 | 54 | 38.64% | 38.64% | 38.64% | 11.36 pp | -20 | 7 | -2.86 |
| BTC Market Hours Daily | xgb | XGBoost | 87 | 32 | 55 | 36.78% | 36.78% | 36.78% | 13.22 pp | -23 | 8 | -2.88 |
| BTC Hourly | rf | RandomForest | 64 | 26 | 38 | 40.62% | 40.62% | 40.62% | 9.38 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 90 | 34 | 56 | 37.78% | 37.78% | 37.78% | 12.22 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 5 | -5.20 |
| BTC Hourly | lstm | LSTM | 64 | 23 | 41 | 35.94% | 35.94% | 35.94% | 14.06 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 100 | 31 | 69 | 31.00% | 31.00% | 31.00% | 19.00 pp | -38 | 6 | -6.33 |
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
| BTC Daily | mlp_sklearn | MLPClassifier | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 5 | -1.60 |
| BTC Daily | rf | RandomForest | 90 | 34 | 56 | 37.78% | 37.78% | 37.78% | 12.22 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 90 | 32 | 58 | 35.56% | 35.56% | 35.56% | 14.44 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 100 | 31 | 69 | 31.00% | 31.00% | 31.00% | 19.00 pp | -38 | 6 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 88 | 50 | 38 | 56.82% | 56.82% | 56.82% | 6.82 pp | 12 | 7 | 1.71 |
| BTC Market Hours | rf | RandomForest | 88 | 44 | 44 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 88 | 43 | 45 | 48.86% | 48.86% | 48.86% | 1.14 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours | transformer | Transformer | 88 | 37 | 51 | 42.05% | 42.05% | 42.05% | 7.95 pp | -14 | 7 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 88 | 34 | 54 | 38.64% | 38.64% | 38.64% | 11.36 pp | -20 | 7 | -2.86 |

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
