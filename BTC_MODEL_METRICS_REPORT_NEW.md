# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T13:35:12.440586+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 129 | 69 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 165 | 105 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 185 | 93 | 92 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 12:00:00+00:00 | 184 | 92 | 92 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 01:00:00+00:00 | 72 | 0 | 72 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| BTC Hourly | transformer | Transformer | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Hourly | nn | NN | 69 | 34 | 35 | 49.28% | 49.28% | 49.28% | 0.72 pp | -1 | 3 | -0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Market Hours Daily | rf | RandomForest | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| BTC Daily | nn | NN | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 92 | 40 | 52 | 43.48% | 43.48% | 43.48% | 6.52 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| BTC Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 3 | -2.33 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |
| BTC Market Hours Daily | xgb | XGBoost | 92 | 35 | 57 | 38.04% | 38.04% | 38.04% | 11.96 pp | -22 | 8 | -2.75 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Hourly | rf | RandomForest | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Hourly | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 3 | -5.67 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |
| BTC Hourly | xgb | XGBoost | 69 | 21 | 48 | 30.43% | 30.43% | 30.43% | 19.57 pp | -27 | 3 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 69 | 36 | 33 | 52.17% | 52.17% | 52.17% | 2.17 pp | 3 | 3 | 1.00 |
| BTC Hourly | nn | NN | 69 | 34 | 35 | 49.28% | 49.28% | 49.28% | 0.72 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 69 | 31 | 38 | 44.93% | 44.93% | 44.93% | 5.07 pp | -7 | 3 | -2.33 |
| BTC Hourly | rf | RandomForest | 69 | 28 | 41 | 40.58% | 40.58% | 40.58% | 9.42 pp | -13 | 3 | -4.33 |
| BTC Hourly | lstm | LSTM | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 3 | -5.67 |
| BTC Hourly | xgb | XGBoost | 69 | 21 | 48 | 30.43% | 30.43% | 30.43% | 19.57 pp | -27 | 3 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 95 | 45 | 50 | 47.37% | 47.37% | 47.37% | 2.63 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 95 | 44 | 51 | 46.32% | 46.32% | 46.32% | 3.68 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 95 | 43 | 52 | 45.26% | 45.26% | 45.26% | 4.74 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 95 | 34 | 61 | 35.79% | 35.79% | 35.79% | 14.21 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 105 | 35 | 70 | 33.33% | 33.33% | 33.33% | 16.67 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 93 | 51 | 42 | 54.84% | 54.84% | 54.84% | 4.84 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 93 | 45 | 48 | 48.39% | 48.39% | 48.39% | 1.61 pp | -3 | 8 | -0.38 |
| BTC Market Hours | lstm | LSTM | 93 | 39 | 54 | 41.94% | 41.94% | 41.94% | 8.06 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 93 | 37 | 56 | 39.78% | 39.78% | 39.78% | 10.22 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 8 | -2.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | nn | NN | 92 | 40 | 52 | 43.48% | 43.48% | 43.48% | 6.52 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 92 | 35 | 57 | 38.04% | 38.04% | 38.04% | 11.96 pp | -22 | 8 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

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
