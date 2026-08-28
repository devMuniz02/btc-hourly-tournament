# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T12:47:46.847078+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 31 | 71 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 127 | 67 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 120 | 55 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 120 | 55 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 2 | 1.50 |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| BTC Market Hours | nn | NN | 55 | 30 | 25 | 54.55% | 54.55% | 54.55% | 4.55 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| BTC Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Hourly | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 6 | -0.50 |
| BTC Market Hours | rf | RandomForest | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| BTC Daily | nn | NN | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| BTC Hourly | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 2 | -1.50 |
| BTC Market Hours Daily | nn | NN | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 6 | -2.17 |
| BTC Hourly | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Market Hours | lstm | LSTM | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |
| BTC Market Hours Daily | lstm | LSTM | 55 | 17 | 38 | 30.91% | 30.91% | 30.91% | 19.09 pp | -21 | 6 | -3.50 |
| BTC Daily | rf | RandomForest | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 3 | -3.67 |
| BTC Daily | xgb | XGBoost | 67 | 23 | 44 | 34.33% | 34.33% | 34.33% | 15.67 pp | -21 | 4 | -5.25 |
| BTC Hourly | rf | RandomForest | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |
| BTC Daily | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 3 | -5.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 31 | 17 | 14 | 54.84% | 54.84% | 54.84% | 4.84 pp | 3 | 2 | 1.50 |
| BTC Hourly | nn | NN | 31 | 15 | 16 | 48.39% | 48.39% | 48.39% | 1.61 pp | -1 | 2 | -0.50 |
| BTC Hourly | lstm | LSTM | 31 | 14 | 17 | 45.16% | 45.16% | 45.16% | 4.84 pp | -3 | 2 | -1.50 |
| BTC Hourly | transformer | Transformer | 31 | 13 | 18 | 41.94% | 41.94% | 41.94% | 8.06 pp | -5 | 2 | -2.50 |
| BTC Hourly | rf | RandomForest | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |
| BTC Hourly | xgb | XGBoost | 31 | 10 | 21 | 32.26% | 32.26% | 32.26% | 17.74 pp | -11 | 2 | -5.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Daily | transformer | Transformer | 57 | 28 | 29 | 49.12% | 49.12% | 49.12% | 0.88 pp | -1 | 3 | -0.33 |
| BTC Daily | nn | NN | 57 | 27 | 30 | 47.37% | 47.37% | 47.37% | 2.63 pp | -3 | 3 | -1.00 |
| BTC Daily | rf | RandomForest | 57 | 23 | 34 | 40.35% | 40.35% | 40.35% | 9.65 pp | -11 | 3 | -3.67 |
| BTC Daily | xgb | XGBoost | 67 | 23 | 44 | 34.33% | 34.33% | 34.33% | 15.67 pp | -21 | 4 | -5.25 |
| BTC Daily | lstm | LSTM | 57 | 20 | 37 | 35.09% | 35.09% | 35.09% | 14.91 pp | -17 | 3 | -5.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 55 | 30 | 25 | 54.55% | 54.55% | 54.55% | 4.55 pp | 5 | 5 | 1.00 |
| BTC Market Hours | rf | RandomForest | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 5 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 55 | 24 | 31 | 43.64% | 43.64% | 43.64% | 6.36 pp | -7 | 5 | -1.40 |
| BTC Market Hours | xgb | XGBoost | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 5 | -2.60 |
| BTC Market Hours | lstm | LSTM | 55 | 19 | 36 | 34.55% | 34.55% | 34.55% | 15.45 pp | -17 | 5 | -3.40 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 55 | 27 | 28 | 49.09% | 49.09% | 49.09% | 0.91 pp | -1 | 6 | -0.17 |
| BTC Market Hours Daily | transformer | Transformer | 55 | 26 | 29 | 47.27% | 47.27% | 47.27% | 2.73 pp | -3 | 6 | -0.50 |
| BTC Market Hours Daily | nn | NN | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | rf | RandomForest | 55 | 22 | 33 | 40.00% | 40.00% | 40.00% | 10.00 pp | -11 | 6 | -1.83 |
| BTC Market Hours Daily | xgb | XGBoost | 55 | 21 | 34 | 38.18% | 38.18% | 38.18% | 11.82 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 55 | 17 | 38 | 30.91% | 30.91% | 30.91% | 19.09 pp | -21 | 6 | -3.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

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
