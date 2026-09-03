# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-03T15:05:08.801348+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 196 | 136 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 232 | 172 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-03 14:00:00+00:00 | 306 | 160 | 146 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-03 14:00:00+00:00 | 306 | 160 | 146 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| BTC Market Hours | nn | NN | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 13 | 0.46 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 136 | 69 | 67 | 50.74% | 50.74% | 50.74% | 0.74 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours Daily | transformer | Transformer | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 14 | -0.29 |
| BTC Hourly | transformer | Transformer | 136 | 67 | 69 | 49.26% | 49.26% | 49.26% | 0.74 pp | -2 | 6 | -0.33 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| BTC Daily | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 8 | -0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 14 | -0.57 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| BTC Market Hours | rf | RandomForest | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours Daily | nn | NN | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 14 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 13 | -1.23 |
| BTC Daily | nn | NN | 162 | 76 | 86 | 46.91% | 46.91% | 46.91% | 3.09 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 14 | -1.57 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Market Hours Daily | xgb | XGBoost | 160 | 66 | 94 | 41.25% | 41.25% | 41.25% | 8.75 pp | -28 | 14 | -2.00 |
| BTC Daily | transformer | Transformer | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 8 | -2.25 |
| BTC Market Hours | lstm | LSTM | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| BTC Market Hours | xgb | XGBoost | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| BTC Hourly | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 160 | 62 | 98 | 38.75% | 38.75% | 38.75% | 11.25 pp | -36 | 14 | -2.57 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Daily | rf | RandomForest | 162 | 69 | 93 | 42.59% | 42.59% | 42.59% | 7.41 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 6 | -3.33 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 172 | 64 | 108 | 37.21% | 37.21% | 37.21% | 12.79 pp | -44 | 9 | -4.89 |
| BTC Daily | lstm | LSTM | 162 | 61 | 101 | 37.65% | 37.65% | 37.65% | 12.35 pp | -40 | 8 | -5.00 |
| BTC Hourly | xgb | XGBoost | 136 | 51 | 85 | 37.50% | 37.50% | 37.50% | 12.50 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 136 | 49 | 87 | 36.03% | 36.03% | 36.03% | 13.97 pp | -38 | 6 | -6.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 136 | 69 | 67 | 50.74% | 50.74% | 50.74% | 0.74 pp | 2 | 6 | 0.33 |
| BTC Hourly | transformer | Transformer | 136 | 67 | 69 | 49.26% | 49.26% | 49.26% | 0.74 pp | -2 | 6 | -0.33 |
| BTC Hourly | nn | NN | 136 | 61 | 75 | 44.85% | 44.85% | 44.85% | 5.15 pp | -14 | 6 | -2.33 |
| BTC Hourly | rf | RandomForest | 136 | 58 | 78 | 42.65% | 42.65% | 42.65% | 7.35 pp | -20 | 6 | -3.33 |
| BTC Hourly | xgb | XGBoost | 136 | 51 | 85 | 37.50% | 37.50% | 37.50% | 12.50 pp | -34 | 6 | -5.67 |
| BTC Hourly | lstm | LSTM | 136 | 49 | 87 | 36.03% | 36.03% | 36.03% | 13.97 pp | -38 | 6 | -6.33 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 162 | 79 | 83 | 48.77% | 48.77% | 48.77% | 1.23 pp | -4 | 8 | -0.50 |
| BTC Daily | nn | NN | 162 | 76 | 86 | 46.91% | 46.91% | 46.91% | 3.09 pp | -10 | 8 | -1.25 |
| BTC Daily | transformer | Transformer | 162 | 72 | 90 | 44.44% | 44.44% | 44.44% | 5.56 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 162 | 69 | 93 | 42.59% | 42.59% | 42.59% | 7.41 pp | -24 | 8 | -3.00 |
| BTC Daily | xgb | XGBoost | 172 | 64 | 108 | 37.21% | 37.21% | 37.21% | 12.79 pp | -44 | 9 | -4.89 |
| BTC Daily | lstm | LSTM | 162 | 61 | 101 | 37.65% | 37.65% | 37.65% | 12.35 pp | -40 | 8 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 13 | 0.46 |
| BTC Market Hours | rf | RandomForest | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | transformer | Transformer | 160 | 73 | 87 | 45.62% | 45.62% | 45.62% | 4.38 pp | -14 | 13 | -1.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 13 | -1.23 |
| BTC Market Hours | lstm | LSTM | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |
| BTC Market Hours | xgb | XGBoost | 160 | 65 | 95 | 40.62% | 40.62% | 40.62% | 9.38 pp | -30 | 13 | -2.31 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 160 | 78 | 82 | 48.75% | 48.75% | 48.75% | 1.25 pp | -4 | 14 | -0.29 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 160 | 76 | 84 | 47.50% | 47.50% | 47.50% | 2.50 pp | -8 | 14 | -0.57 |
| BTC Market Hours Daily | nn | NN | 160 | 72 | 88 | 45.00% | 45.00% | 45.00% | 5.00 pp | -16 | 14 | -1.14 |
| BTC Market Hours Daily | rf | RandomForest | 160 | 69 | 91 | 43.12% | 43.12% | 43.12% | 6.87 pp | -22 | 14 | -1.57 |
| BTC Market Hours Daily | xgb | XGBoost | 160 | 66 | 94 | 41.25% | 41.25% | 41.25% | 8.75 pp | -28 | 14 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 160 | 62 | 98 | 38.75% | 38.75% | 38.75% | 11.25 pp | -36 | 14 | -2.57 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
