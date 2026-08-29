# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T22:02:03.280844+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 117 | 57 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 152 | 92 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 168 | 80 | 88 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 168 | 80 | 88 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 62 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 62 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 0 | 62 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 0 | 62 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 80 | 46 | 34 | 57.50% | 57.50% | 57.50% | 7.50 pp | 12 | 7 | 1.71 |
| Consolidated Hourly | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 8 | 0.50 |
| BTC Hourly | transformer | Transformer | 57 | 29 | 28 | 50.88% | 50.88% | 50.88% | 0.88 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| BTC Market Hours Daily | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| Consolidated Hourly | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| BTC Hourly | nn | NN | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 8 | -1.25 |
| BTC Market Hours | lstm | LSTM | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| BTC Daily | mlp_sklearn | MLPClassifier | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 4 | -1.50 |
| BTC Daily | nn | NN | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 4 | -1.50 |
| BTC Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Market Hours Daily | lstm | LSTM | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 7 | -2.29 |
| Consolidated Hourly | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |
| BTC Hourly | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 3 | -4.33 |
| BTC Daily | rf | RandomForest | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 4 | -5.00 |
| BTC Daily | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 57 | 19 | 38 | 33.33% | 33.33% | 33.33% | 16.67 pp | -19 | 3 | -6.33 |
| BTC Daily | xgb | XGBoost | 92 | 27 | 65 | 29.35% | 29.35% | 29.35% | 20.65 pp | -38 | 5 | -7.60 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 57 | 29 | 28 | 50.88% | 50.88% | 50.88% | 0.88 pp | 1 | 3 | 0.33 |
| BTC Hourly | nn | NN | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| BTC Hourly | rf | RandomForest | 57 | 24 | 33 | 42.11% | 42.11% | 42.11% | 7.89 pp | -9 | 3 | -3.00 |
| BTC Hourly | lstm | LSTM | 57 | 22 | 35 | 38.60% | 38.60% | 38.60% | 11.40 pp | -13 | 3 | -4.33 |
| BTC Hourly | xgb | XGBoost | 57 | 19 | 38 | 33.33% | 33.33% | 33.33% | 16.67 pp | -19 | 3 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 82 | 39 | 43 | 47.56% | 47.56% | 47.56% | 2.44 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 4 | -1.50 |
| BTC Daily | nn | NN | 82 | 38 | 44 | 46.34% | 46.34% | 46.34% | 3.66 pp | -6 | 4 | -1.50 |
| BTC Daily | rf | RandomForest | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 4 | -5.00 |
| BTC Daily | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |
| BTC Daily | xgb | XGBoost | 92 | 27 | 65 | 29.35% | 29.35% | 29.35% | 20.65 pp | -38 | 5 | -7.60 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 80 | 46 | 34 | 57.50% | 57.50% | 57.50% | 7.50 pp | 12 | 7 | 1.71 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | rf | RandomForest | 80 | 39 | 41 | 48.75% | 48.75% | 48.75% | 1.25 pp | -2 | 7 | -0.29 |
| BTC Market Hours | lstm | LSTM | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 7 | -1.43 |
| BTC Market Hours | transformer | Transformer | 80 | 34 | 46 | 42.50% | 42.50% | 42.50% | 7.50 pp | -12 | 7 | -1.71 |
| BTC Market Hours | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 7 | -2.29 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 80 | 42 | 38 | 52.50% | 52.50% | 52.50% | 2.50 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 80 | 40 | 40 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 80 | 36 | 44 | 45.00% | 45.00% | 45.00% | 5.00 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 80 | 35 | 45 | 43.75% | 43.75% | 43.75% | 6.25 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 80 | 33 | 47 | 41.25% | 41.25% | 41.25% | 8.75 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | xgb | XGBoost | 80 | 32 | 48 | 40.00% | 40.00% | 40.00% | 10.00 pp | -16 | 8 | -2.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Hourly | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |

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
