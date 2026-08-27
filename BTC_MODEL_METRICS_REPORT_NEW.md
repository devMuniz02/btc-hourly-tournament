# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T23:18:55.788297+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 20 | 82 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 116 | 56 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 107 | 44 | 63 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 22:00:00+00:00 | 107 | 44 | 63 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 33 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 1 | 32 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 16:00:00+00:00 | 33 | 1 | 32 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | rf | RandomForest | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 5 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | nn | NN | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |
| BTC Daily | xgb | XGBoost | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Market Hours | lstm | LSTM | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 44 | 11 | 33 | 25.00% | 25.00% | 25.00% | 25.00 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 3 | -5.33 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | lstm | LSTM | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 20 | 10 | 10 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Hourly | nn | NN | 20 | 9 | 11 | 45.00% | 45.00% | 45.00% | 5.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | transformer | Transformer | 20 | 8 | 12 | 40.00% | 40.00% | 40.00% | 10.00 pp | -4 | 1 | -4.00 |
| BTC Hourly | xgb | XGBoost | 20 | 6 | 14 | 30.00% | 30.00% | 30.00% | 20.00 pp | -8 | 1 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 46 | 24 | 22 | 52.17% | 52.17% | 52.17% | 2.17 pp | 2 | 3 | 0.67 |
| BTC Daily | nn | NN | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 56 | 21 | 35 | 37.50% | 37.50% | 37.50% | 12.50 pp | -14 | 4 | -3.50 |
| BTC Daily | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 3 | -5.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 44 | 24 | 20 | 54.55% | 54.55% | 54.55% | 4.55 pp | 4 | 4 | 1.00 |
| BTC Market Hours | rf | RandomForest | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 44 | 20 | 24 | 45.45% | 45.45% | 45.45% | 4.55 pp | -4 | 4 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 4 | -2.00 |
| BTC Market Hours | lstm | LSTM | 44 | 14 | 30 | 31.82% | 31.82% | 31.82% | 18.18 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 44 | 22 | 22 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 44 | 21 | 23 | 47.73% | 47.73% | 47.73% | 2.27 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | xgb | XGBoost | 44 | 19 | 25 | 43.18% | 43.18% | 43.18% | 6.82 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | nn | NN | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 44 | 18 | 26 | 40.91% | 40.91% | 40.91% | 9.09 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 44 | 11 | 33 | 25.00% | 25.00% | 25.00% | 25.00 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 33 | 19 | 14 | 57.58% | 57.58% | 57.58% | 7.58 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 33 | 18 | 15 | 54.55% | 54.55% | 54.55% | 4.55 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 33 | 16 | 17 | 48.48% | 48.48% | 48.48% | 1.52 pp | -1 | 4 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 33 | 15 | 18 | 45.45% | 45.45% | 45.45% | 4.55 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 33 | 14 | 19 | 42.42% | 42.42% | 42.42% | 7.58 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 33 | 11 | 22 | 33.33% | 33.33% | 33.33% | 16.67 pp | -11 | 4 | -2.75 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
