# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-29T16:55:44.243972+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 113 | 53 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 149 | 89 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 159 | 77 | 82 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-29 15:00:00+00:00 | 158 | 76 | 82 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 60 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 60 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 0 | 60 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 11:00:00+00:00 | 60 | 0 | 60 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 77 | 45 | 32 | 58.44% | 58.44% | 58.44% | 8.44 pp | 13 | 6 | 2.17 |
| Consolidated Hourly | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 7 | 0.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 7 | -0.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 79 | 38 | 41 | 48.10% | 48.10% | 48.10% | 1.90 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 79 | 38 | 41 | 48.10% | 48.10% | 48.10% | 1.90 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 7 | -1.14 |
| BTC Daily | nn | NN | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | nn | NN | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 7 | -1.43 |
| BTC Market Hours | lstm | LSTM | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Hourly | nn | NN | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 3 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 7 | -2.00 |
| BTC Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 7 | -2.29 |
| BTC Market Hours | xgb | XGBoost | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |
| Consolidated Hourly | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |
| BTC Hourly | lstm | LSTM | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Hourly | xgb | XGBoost | 53 | 17 | 36 | 32.08% | 32.08% | 32.08% | 17.92 pp | -19 | 3 | -6.33 |
| BTC Daily | xgb | XGBoost | 89 | 27 | 62 | 30.34% | 30.34% | 30.34% | 19.66 pp | -35 | 5 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 3 | -0.33 |
| BTC Hourly | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 3 | -1.00 |
| BTC Hourly | nn | NN | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 3 | -1.67 |
| BTC Hourly | lstm | LSTM | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 3 | -3.67 |
| BTC Hourly | rf | RandomForest | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 3 | -3.67 |
| BTC Hourly | xgb | XGBoost | 53 | 17 | 36 | 32.08% | 32.08% | 32.08% | 17.92 pp | -19 | 3 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 79 | 38 | 41 | 48.10% | 48.10% | 48.10% | 1.90 pp | -3 | 4 | -0.75 |
| BTC Daily | transformer | Transformer | 79 | 38 | 41 | 48.10% | 48.10% | 48.10% | 1.90 pp | -3 | 4 | -0.75 |
| BTC Daily | nn | NN | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 4 | -1.25 |
| BTC Daily | rf | RandomForest | 79 | 30 | 49 | 37.97% | 37.97% | 37.97% | 12.03 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 79 | 27 | 52 | 34.18% | 34.18% | 34.18% | 15.82 pp | -25 | 4 | -6.25 |
| BTC Daily | xgb | XGBoost | 89 | 27 | 62 | 30.34% | 30.34% | 30.34% | 19.66 pp | -35 | 5 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 77 | 45 | 32 | 58.44% | 58.44% | 58.44% | 8.44 pp | 13 | 6 | 2.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 6 | -0.17 |
| BTC Market Hours | rf | RandomForest | 77 | 38 | 39 | 49.35% | 49.35% | 49.35% | 0.65 pp | -1 | 6 | -0.17 |
| BTC Market Hours | lstm | LSTM | 77 | 34 | 43 | 44.16% | 44.16% | 44.16% | 5.84 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 77 | 32 | 45 | 41.56% | 41.56% | 41.56% | 8.44 pp | -13 | 6 | -2.17 |
| BTC Market Hours | xgb | XGBoost | 77 | 31 | 46 | 40.26% | 40.26% | 40.26% | 9.74 pp | -15 | 6 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 7 | 0.57 |
| BTC Market Hours Daily | transformer | Transformer | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | rf | RandomForest | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 7 | -1.14 |
| BTC Market Hours Daily | nn | NN | 76 | 33 | 43 | 43.42% | 43.42% | 43.42% | 6.58 pp | -10 | 7 | -1.43 |
| BTC Market Hours Daily | lstm | LSTM | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 76 | 30 | 46 | 39.47% | 39.47% | 39.47% | 10.53 pp | -16 | 7 | -2.29 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 60 | 34 | 26 | 56.67% | 56.67% | 56.67% | 6.67 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 60 | 30 | 30 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 6 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 60 | 21 | 39 | 35.00% | 35.00% | 35.00% | 15.00 pp | -18 | 6 | -3.00 |

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
