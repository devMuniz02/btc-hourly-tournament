# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T10:42:58.021320+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 29 | 73 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 125 | 65 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 118 | 53 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 118 | 53 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T01:00:00+00:00 | 39 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T01:00:00+00:00 | 39 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T01:00:00+00:00 | 39 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T01:00:00+00:00 | 40 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 2 | 2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| BTC Market Hours | nn | NN | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 5 | 1.00 |
| BTC Hourly | nn | NN | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 3 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 5 | 0.20 |
| BTC Daily | transformer | Transformer | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Hourly | lstm | LSTM | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 2 | -0.50 |
| BTC Market Hours | rf | RandomForest | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| BTC Market Hours | transformer | Transformer | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| BTC Hourly | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | nn | NN | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours | xgb | XGBoost | 53 | 20 | 33 | 37.74% | 37.74% | 37.74% | 12.26 pp | -13 | 5 | -2.60 |
| BTC Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 3 | -3.00 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |
| BTC Market Hours | lstm | LSTM | 53 | 18 | 35 | 33.96% | 33.96% | 33.96% | 16.04 pp | -17 | 5 | -3.40 |
| BTC Market Hours Daily | lstm | LSTM | 53 | 16 | 37 | 30.19% | 30.19% | 30.19% | 19.81 pp | -21 | 5 | -4.20 |
| BTC Hourly | rf | RandomForest | 29 | 10 | 19 | 34.48% | 34.48% | 34.48% | 15.52 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 29 | 10 | 19 | 34.48% | 34.48% | 34.48% | 15.52 pp | -9 | 2 | -4.50 |
| BTC Daily | xgb | XGBoost | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 55 | 18 | 37 | 32.73% | 32.73% | 32.73% | 17.27 pp | -19 | 3 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 29 | 17 | 12 | 58.62% | 58.62% | 58.62% | 8.62 pp | 5 | 2 | 2.50 |
| BTC Hourly | nn | NN | 29 | 15 | 14 | 51.72% | 51.72% | 51.72% | 1.72 pp | 1 | 2 | 0.50 |
| BTC Hourly | lstm | LSTM | 29 | 14 | 15 | 48.28% | 48.28% | 48.28% | 1.72 pp | -1 | 2 | -0.50 |
| BTC Hourly | transformer | Transformer | 29 | 13 | 16 | 44.83% | 44.83% | 44.83% | 5.17 pp | -3 | 2 | -1.50 |
| BTC Hourly | rf | RandomForest | 29 | 10 | 19 | 34.48% | 34.48% | 34.48% | 15.52 pp | -9 | 2 | -4.50 |
| BTC Hourly | xgb | XGBoost | 29 | 10 | 19 | 34.48% | 34.48% | 34.48% | 15.52 pp | -9 | 2 | -4.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 55 | 28 | 27 | 50.91% | 50.91% | 50.91% | 0.91 pp | 1 | 3 | 0.33 |
| BTC Daily | transformer | Transformer | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 55 | 23 | 32 | 41.82% | 41.82% | 41.82% | 8.18 pp | -9 | 3 | -3.00 |
| BTC Daily | xgb | XGBoost | 65 | 23 | 42 | 35.38% | 35.38% | 35.38% | 14.62 pp | -19 | 4 | -4.75 |
| BTC Daily | lstm | LSTM | 55 | 18 | 37 | 32.73% | 32.73% | 32.73% | 17.27 pp | -19 | 3 | -6.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 53 | 29 | 24 | 54.72% | 54.72% | 54.72% | 4.72 pp | 5 | 5 | 1.00 |
| BTC Market Hours | rf | RandomForest | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| BTC Market Hours | transformer | Transformer | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 53 | 20 | 33 | 37.74% | 37.74% | 37.74% | 12.26 pp | -13 | 5 | -2.60 |
| BTC Market Hours | lstm | LSTM | 53 | 18 | 35 | 33.96% | 33.96% | 33.96% | 16.04 pp | -17 | 5 | -3.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 27 | 26 | 50.94% | 50.94% | 50.94% | 0.94 pp | 1 | 5 | 0.20 |
| BTC Market Hours Daily | transformer | Transformer | 53 | 25 | 28 | 47.17% | 47.17% | 47.17% | 2.83 pp | -3 | 5 | -0.60 |
| BTC Market Hours Daily | nn | NN | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 53 | 16 | 37 | 30.19% | 30.19% | 30.19% | 19.81 pp | -21 | 5 | -4.20 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 39 | 18 | 21 | 46.15% | 46.15% | 46.15% | 3.85 pp | -3 | 5 | -0.60 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 39 | 16 | 23 | 41.03% | 41.03% | 41.03% | 8.97 pp | -7 | 5 | -1.40 |
| Consolidated Daily/Hourly Refresh | nn | NN | 39 | 12 | 27 | 30.77% | 30.77% | 30.77% | 19.23 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
