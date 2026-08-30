# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T11:21:30.975378+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 127 | 67 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 163 | 103 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 182 | 91 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 182 | 91 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 00:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 00:00:00+00:00 | 72 | 72 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 00:00:00+00:00 | 72 | 1 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 00:00:00+00:00 | 72 | 1 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 91 | 51 | 40 | 56.04% | 56.04% | 56.04% | 6.04 pp | 11 | 7 | 1.57 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 8 | 0.38 |
| BTC Hourly | nn | NN | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 3 | 0.33 |
| BTC Market Hours | rf | RandomForest | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 7 | -0.14 |
| BTC Market Hours Daily | transformer | Transformer | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | rf | RandomForest | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 8 | -0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 8 | -1.38 |
| BTC Daily | nn | NN | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 3 | -1.67 |
| BTC Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Market Hours Daily | lstm | LSTM | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 8 | -2.12 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| BTC Market Hours | transformer | Transformer | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |
| BTC Market Hours Daily | xgb | XGBoost | 91 | 34 | 57 | 37.36% | 37.36% | 37.36% | 12.64 pp | -23 | 8 | -2.88 |
| BTC Hourly | rf | RandomForest | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 3 | -3.67 |
| BTC Daily | rf | RandomForest | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 103 | 34 | 69 | 33.01% | 33.01% | 33.01% | 16.99 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 67 | 21 | 46 | 31.34% | 31.34% | 31.34% | 18.66 pp | -25 | 3 | -8.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 3 | 0.33 |
| BTC Hourly | transformer | Transformer | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 3 | 0.33 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 67 | 31 | 36 | 46.27% | 46.27% | 46.27% | 3.73 pp | -5 | 3 | -1.67 |
| BTC Hourly | rf | RandomForest | 67 | 28 | 39 | 41.79% | 41.79% | 41.79% | 8.21 pp | -11 | 3 | -3.67 |
| BTC Hourly | lstm | LSTM | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 3 | -6.33 |
| BTC Hourly | xgb | XGBoost | 67 | 21 | 46 | 31.34% | 31.34% | 31.34% | 18.66 pp | -25 | 3 | -8.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 5 | -1.00 |
| BTC Daily | nn | NN | 93 | 43 | 50 | 46.24% | 46.24% | 46.24% | 3.76 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 93 | 42 | 51 | 45.16% | 45.16% | 45.16% | 4.84 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 93 | 36 | 57 | 38.71% | 38.71% | 38.71% | 11.29 pp | -21 | 5 | -4.20 |
| BTC Daily | lstm | LSTM | 93 | 33 | 60 | 35.48% | 35.48% | 35.48% | 14.52 pp | -27 | 5 | -5.40 |
| BTC Daily | xgb | XGBoost | 103 | 34 | 69 | 33.01% | 33.01% | 33.01% | 16.99 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 91 | 51 | 40 | 56.04% | 56.04% | 56.04% | 6.04 pp | 11 | 7 | 1.57 |
| BTC Market Hours | rf | RandomForest | 91 | 46 | 45 | 50.55% | 50.55% | 50.55% | 0.55 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 7 | -0.14 |
| BTC Market Hours | lstm | LSTM | 91 | 39 | 52 | 42.86% | 42.86% | 42.86% | 7.14 pp | -13 | 7 | -1.86 |
| BTC Market Hours | transformer | Transformer | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 7 | -2.43 |
| BTC Market Hours | xgb | XGBoost | 91 | 36 | 55 | 39.56% | 39.56% | 39.56% | 10.44 pp | -19 | 7 | -2.71 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 91 | 47 | 44 | 51.65% | 51.65% | 51.65% | 1.65 pp | 3 | 8 | 0.38 |
| BTC Market Hours Daily | transformer | Transformer | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 8 | -0.62 |
| BTC Market Hours Daily | rf | RandomForest | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 8 | -0.88 |
| BTC Market Hours Daily | nn | NN | 91 | 40 | 51 | 43.96% | 43.96% | 43.96% | 6.04 pp | -11 | 8 | -1.38 |
| BTC Market Hours Daily | lstm | LSTM | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 8 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 91 | 34 | 57 | 37.36% | 37.36% | 37.36% | 12.64 pp | -23 | 8 | -2.88 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 8 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 40 | 32 | 55.56% | 55.56% | 55.56% | 5.56 pp | 8 | 8 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
