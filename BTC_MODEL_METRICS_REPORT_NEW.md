# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T19:19:11.222284+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 133 | 73 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 169 | 109 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 18:00:00+00:00 | 195 | 97 | 98 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 18:00:00+00:00 | 195 | 97 | 98 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 17:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 17:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 17:00:00+00:00 | 78 | 1 | 77 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 17:00:00+00:00 | 78 | 1 | 77 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 97 | 53 | 44 | 54.64% | 54.64% | 54.64% | 4.64 pp | 9 | 8 | 1.12 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Hourly | transformer | Transformer | 73 | 38 | 35 | 52.05% | 52.05% | 52.05% | 2.05 pp | 3 | 3 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | lstm | LSTM | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | nn | NN | 73 | 36 | 37 | 49.32% | 49.32% | 49.32% | 0.68 pp | -1 | 3 | -0.33 |
| BTC Market Hours | rf | RandomForest | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 8 | -0.62 |
| BTC Daily | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 9 | -1.67 |
| BTC Daily | nn | NN | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 8 | -1.88 |
| Consolidated Hourly | nn | NN | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 8 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 99 | 44 | 55 | 44.44% | 44.44% | 44.44% | 5.56 pp | -11 | 5 | -2.20 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 3 | -2.33 |
| BTC Market Hours | transformer | Transformer | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 8 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 9 | -2.78 |
| BTC Market Hours | xgb | XGBoost | 97 | 37 | 60 | 38.14% | 38.14% | 38.14% | 11.86 pp | -23 | 8 | -2.88 |
| BTC Daily | rf | RandomForest | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 5 | -4.20 |
| BTC Hourly | rf | RandomForest | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 3 | -5.00 |
| BTC Daily | lstm | LSTM | 99 | 36 | 63 | 36.36% | 36.36% | 36.36% | 13.64 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 73 | 26 | 47 | 35.62% | 35.62% | 35.62% | 14.38 pp | -21 | 3 | -7.00 |
| BTC Hourly | xgb | XGBoost | 73 | 24 | 49 | 32.88% | 32.88% | 32.88% | 17.12 pp | -25 | 3 | -8.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 73 | 38 | 35 | 52.05% | 52.05% | 52.05% | 2.05 pp | 3 | 3 | 1.00 |
| BTC Hourly | nn | NN | 73 | 36 | 37 | 49.32% | 49.32% | 49.32% | 0.68 pp | -1 | 3 | -0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 73 | 33 | 40 | 45.21% | 45.21% | 45.21% | 4.79 pp | -7 | 3 | -2.33 |
| BTC Hourly | rf | RandomForest | 73 | 29 | 44 | 39.73% | 39.73% | 39.73% | 10.27 pp | -15 | 3 | -5.00 |
| BTC Hourly | lstm | LSTM | 73 | 26 | 47 | 35.62% | 35.62% | 35.62% | 14.38 pp | -21 | 3 | -7.00 |
| BTC Hourly | xgb | XGBoost | 73 | 24 | 49 | 32.88% | 32.88% | 32.88% | 17.12 pp | -25 | 3 | -8.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 5 | -1.80 |
| BTC Daily | transformer | Transformer | 99 | 44 | 55 | 44.44% | 44.44% | 44.44% | 5.56 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 99 | 36 | 63 | 36.36% | 36.36% | 36.36% | 13.64 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 6 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 97 | 53 | 44 | 54.64% | 54.64% | 54.64% | 4.64 pp | 9 | 8 | 1.12 |
| BTC Market Hours | rf | RandomForest | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 8 | -0.38 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 97 | 46 | 51 | 47.42% | 47.42% | 47.42% | 2.58 pp | -5 | 8 | -0.62 |
| BTC Market Hours | lstm | LSTM | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 97 | 37 | 60 | 38.14% | 38.14% | 38.14% | 11.86 pp | -23 | 8 | -2.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 97 | 49 | 48 | 50.52% | 50.52% | 50.52% | 0.52 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | rf | RandomForest | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 97 | 44 | 53 | 45.36% | 45.36% | 45.36% | 4.64 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 97 | 41 | 56 | 42.27% | 42.27% | 42.27% | 7.73 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 97 | 36 | 61 | 37.11% | 37.11% | 37.11% | 12.89 pp | -25 | 9 | -2.78 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | nn | NN | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 8 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 39 | 39 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 34 | 44 | 43.59% | 43.59% | 43.59% | 6.41 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 31 | 47 | 39.74% | 39.74% | 39.74% | 10.26 pp | -16 | 8 | -2.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
