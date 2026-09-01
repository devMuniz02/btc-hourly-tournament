# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T22:58:37.872291+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 169 | 109 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 204 | 144 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 21:00:00+00:00 | 259 | 132 | 127 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 21:00:00+00:00 | 259 | 132 | 127 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 109 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 12:00:00+00:00 | 109 | 15 | 94 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| BTC Market Hours | nn | NN | 132 | 70 | 62 | 53.03% | 53.03% | 53.03% | 3.03 pp | 8 | 11 | 0.73 |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| BTC Market Hours | rf | RandomForest | 132 | 64 | 68 | 48.48% | 48.48% | 48.48% | 1.52 pp | -4 | 11 | -0.36 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 132 | 63 | 69 | 47.73% | 47.73% | 47.73% | 2.27 pp | -6 | 12 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| BTC Daily | mlp_sklearn | MLPClassifier | 134 | 64 | 70 | 47.76% | 47.76% | 47.76% | 2.24 pp | -6 | 6 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 12 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 12 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | nn | NN | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 12 | -1.17 |
| BTC Hourly | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Market Hours | transformer | Transformer | 132 | 57 | 75 | 43.18% | 43.18% | 43.18% | 6.82 pp | -18 | 11 | -1.64 |
| BTC Market Hours | xgb | XGBoost | 132 | 56 | 76 | 42.42% | 42.42% | 42.42% | 7.58 pp | -20 | 11 | -1.82 |
| BTC Daily | nn | NN | 134 | 61 | 73 | 45.52% | 45.52% | 45.52% | 4.48 pp | -12 | 6 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 132 | 54 | 78 | 40.91% | 40.91% | 40.91% | 9.09 pp | -24 | 12 | -2.00 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| BTC Daily | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 6 | -2.67 |
| BTC Market Hours Daily | lstm | LSTM | 132 | 50 | 82 | 37.88% | 37.88% | 37.88% | 12.12 pp | -32 | 12 | -2.67 |
| BTC Market Hours | lstm | LSTM | 132 | 51 | 81 | 38.64% | 38.64% | 38.64% | 11.36 pp | -30 | 11 | -2.73 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |
| BTC Daily | rf | RandomForest | 134 | 55 | 79 | 41.04% | 41.04% | 41.04% | 8.96 pp | -24 | 6 | -4.00 |
| BTC Hourly | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 7 | -6.00 |
| BTC Hourly | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 5 | -6.60 |
| BTC Daily | lstm | LSTM | 134 | 47 | 87 | 35.07% | 35.07% | 35.07% | 14.93 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 109 | 34 | 75 | 31.19% | 31.19% | 31.19% | 18.81 pp | -41 | 5 | -8.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 109 | 52 | 57 | 47.71% | 47.71% | 47.71% | 2.29 pp | -5 | 5 | -1.00 |
| BTC Hourly | nn | NN | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 5 | -1.40 |
| BTC Hourly | rf | RandomForest | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 5 | -4.20 |
| BTC Hourly | xgb | XGBoost | 109 | 38 | 71 | 34.86% | 34.86% | 34.86% | 15.14 pp | -33 | 5 | -6.60 |
| BTC Hourly | lstm | LSTM | 109 | 34 | 75 | 31.19% | 31.19% | 31.19% | 18.81 pp | -41 | 5 | -8.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 134 | 64 | 70 | 47.76% | 47.76% | 47.76% | 2.24 pp | -6 | 6 | -1.00 |
| BTC Daily | nn | NN | 134 | 61 | 73 | 45.52% | 45.52% | 45.52% | 4.48 pp | -12 | 6 | -2.00 |
| BTC Daily | transformer | Transformer | 134 | 59 | 75 | 44.03% | 44.03% | 44.03% | 5.97 pp | -16 | 6 | -2.67 |
| BTC Daily | rf | RandomForest | 134 | 55 | 79 | 41.04% | 41.04% | 41.04% | 8.96 pp | -24 | 6 | -4.00 |
| BTC Daily | xgb | XGBoost | 144 | 51 | 93 | 35.42% | 35.42% | 35.42% | 14.58 pp | -42 | 7 | -6.00 |
| BTC Daily | lstm | LSTM | 134 | 47 | 87 | 35.07% | 35.07% | 35.07% | 14.93 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 132 | 70 | 62 | 53.03% | 53.03% | 53.03% | 3.03 pp | 8 | 11 | 0.73 |
| BTC Market Hours | rf | RandomForest | 132 | 64 | 68 | 48.48% | 48.48% | 48.48% | 1.52 pp | -4 | 11 | -0.36 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 11 | -1.09 |
| BTC Market Hours | transformer | Transformer | 132 | 57 | 75 | 43.18% | 43.18% | 43.18% | 6.82 pp | -18 | 11 | -1.64 |
| BTC Market Hours | xgb | XGBoost | 132 | 56 | 76 | 42.42% | 42.42% | 42.42% | 7.58 pp | -20 | 11 | -1.82 |
| BTC Market Hours | lstm | LSTM | 132 | 51 | 81 | 38.64% | 38.64% | 38.64% | 11.36 pp | -30 | 11 | -2.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 132 | 63 | 69 | 47.73% | 47.73% | 47.73% | 2.27 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 12 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 132 | 60 | 72 | 45.45% | 45.45% | 45.45% | 4.55 pp | -12 | 12 | -1.00 |
| BTC Market Hours Daily | nn | NN | 132 | 59 | 73 | 44.70% | 44.70% | 44.70% | 5.30 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | xgb | XGBoost | 132 | 54 | 78 | 40.91% | 40.91% | 40.91% | 9.09 pp | -24 | 12 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 132 | 50 | 82 | 37.88% | 37.88% | 37.88% | 12.12 pp | -32 | 12 | -2.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Hourly | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 109 | 57 | 52 | 52.29% | 52.29% | 52.29% | 2.29 pp | 5 | 10 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 109 | 54 | 55 | 49.54% | 49.54% | 49.54% | 0.46 pp | -1 | 10 | -0.10 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 109 | 53 | 56 | 48.62% | 48.62% | 48.62% | 1.38 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 109 | 51 | 58 | 46.79% | 46.79% | 46.79% | 3.21 pp | -7 | 10 | -0.70 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 109 | 49 | 60 | 44.95% | 44.95% | 44.95% | 5.05 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | nn | NN | 109 | 44 | 65 | 40.37% | 40.37% | 40.37% | 9.63 pp | -21 | 10 | -2.10 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 15 | 9 | 6 | 60.00% | 60.00% | 60.00% | 10.00 pp | 3 | 2 | 1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 15 | 7 | 8 | 46.67% | 46.67% | 46.67% | 3.33 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 15 | 5 | 10 | 33.33% | 33.33% | 33.33% | 16.67 pp | -5 | 2 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 15 | 4 | 11 | 26.67% | 26.67% | 26.67% | 23.33 pp | -7 | 2 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
