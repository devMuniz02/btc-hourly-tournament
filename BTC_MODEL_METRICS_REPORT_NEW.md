# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T08:05:10.542575+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 179 | 88 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 21:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 21:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 21:00:00+00:00 | 71 | 1 | 70 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 21:00:00+00:00 | 71 | 1 | 70 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 88 | 50 | 38 | 56.82% | 56.82% | 56.82% | 6.82 pp | 12 | 7 | 1.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| BTC Hourly | nn | NN | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Hourly | transformer | Transformer | 64 | 33 | 31 | 51.56% | 51.56% | 51.56% | 1.56 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 46 | 42 | 52.27% | 52.27% | 52.27% | 2.27 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | rf | RandomForest | 88 | 44 | 44 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 7 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 88 | 43 | 45 | 48.86% | 48.86% | 48.86% | 1.14 pp | -2 | 7 | -0.29 |
| BTC Market Hours Daily | transformer | Transformer | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 88 | 40 | 48 | 45.45% | 45.45% | 45.45% | 4.55 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 90 | 42 | 48 | 46.67% | 46.67% | 46.67% | 3.33 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | nn | NN | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 8 | -1.50 |
| BTC Daily | transformer | Transformer | 90 | 41 | 49 | 45.56% | 45.56% | 45.56% | 4.44 pp | -8 | 5 | -1.60 |
| BTC Market Hours | lstm | LSTM | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 7 | -1.71 |
| BTC Market Hours | transformer | Transformer | 88 | 37 | 51 | 42.05% | 42.05% | 42.05% | 7.95 pp | -14 | 7 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 88 | 36 | 52 | 40.91% | 40.91% | 40.91% | 9.09 pp | -16 | 8 | -2.00 |
| Consolidated Hourly | nn | NN | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 7 | -2.71 |
| BTC Market Hours | xgb | XGBoost | 88 | 34 | 54 | 38.64% | 38.64% | 38.64% | 11.36 pp | -20 | 7 | -2.86 |
| BTC Market Hours Daily | xgb | XGBoost | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 8 | -3.00 |
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
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 88 | 46 | 42 | 52.27% | 52.27% | 52.27% | 2.27 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 88 | 42 | 46 | 47.73% | 47.73% | 47.73% | 2.27 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 88 | 40 | 48 | 45.45% | 45.45% | 45.45% | 4.55 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 88 | 38 | 50 | 43.18% | 43.18% | 43.18% | 6.82 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 88 | 36 | 52 | 40.91% | 40.91% | 40.91% | 9.09 pp | -16 | 8 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 88 | 32 | 56 | 36.36% | 36.36% | 36.36% | 13.64 pp | -24 | 8 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | nn | NN | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 26 | 45 | 36.62% | 36.62% | 36.62% | 13.38 pp | -19 | 7 | -2.71 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
