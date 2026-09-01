# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T14:59:57.404381+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 164 | 104 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 200 | 140 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 247 | 128 | 119 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 13:00:00+00:00 | 246 | 127 | 119 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 105 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 22:00:00+00:00 | 105 | 12 | 93 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| BTC Market Hours | nn | NN | 128 | 68 | 60 | 53.12% | 53.12% | 53.12% | 3.12 pp | 8 | 10 | 0.80 |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 11 | -0.45 |
| BTC Market Hours | rf | RandomForest | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 10 | -0.60 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 104 | 50 | 54 | 48.08% | 48.08% | 48.08% | 1.92 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 11 | -1.18 |
| BTC Hourly | nn | NN | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| BTC Daily | nn | NN | 130 | 61 | 69 | 46.92% | 46.92% | 46.92% | 3.08 pp | -8 | 6 | -1.33 |
| BTC Market Hours | transformer | Transformer | 128 | 55 | 73 | 42.97% | 42.97% | 42.97% | 7.03 pp | -18 | 10 | -1.80 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| BTC Daily | transformer | Transformer | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Market Hours | xgb | XGBoost | 128 | 52 | 76 | 40.62% | 40.62% | 40.62% | 9.38 pp | -24 | 10 | -2.40 |
| BTC Market Hours Daily | xgb | XGBoost | 127 | 50 | 77 | 39.37% | 39.37% | 39.37% | 10.63 pp | -27 | 11 | -2.45 |
| BTC Market Hours | lstm | LSTM | 128 | 49 | 79 | 38.28% | 38.28% | 38.28% | 11.72 pp | -30 | 10 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 11 | -3.00 |
| BTC Hourly | rf | RandomForest | 104 | 43 | 61 | 41.35% | 41.35% | 41.35% | 8.65 pp | -18 | 5 | -3.60 |
| BTC Daily | rf | RandomForest | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 6 | -3.67 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| BTC Daily | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 7 | -5.43 |
| BTC Hourly | xgb | XGBoost | 104 | 36 | 68 | 34.62% | 34.62% | 34.62% | 15.38 pp | -32 | 5 | -6.40 |
| BTC Daily | lstm | LSTM | 130 | 45 | 85 | 34.62% | 34.62% | 34.62% | 15.38 pp | -40 | 6 | -6.67 |
| BTC Hourly | lstm | LSTM | 104 | 32 | 72 | 30.77% | 30.77% | 30.77% | 19.23 pp | -40 | 5 | -8.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 104 | 50 | 54 | 48.08% | 48.08% | 48.08% | 1.92 pp | -4 | 5 | -0.80 |
| BTC Hourly | nn | NN | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | transformer | Transformer | 104 | 49 | 55 | 47.12% | 47.12% | 47.12% | 2.88 pp | -6 | 5 | -1.20 |
| BTC Hourly | rf | RandomForest | 104 | 43 | 61 | 41.35% | 41.35% | 41.35% | 8.65 pp | -18 | 5 | -3.60 |
| BTC Hourly | xgb | XGBoost | 104 | 36 | 68 | 34.62% | 34.62% | 34.62% | 15.38 pp | -32 | 5 | -6.40 |
| BTC Hourly | lstm | LSTM | 104 | 32 | 72 | 30.77% | 30.77% | 30.77% | 19.23 pp | -40 | 5 | -8.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 130 | 65 | 65 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Daily | nn | NN | 130 | 61 | 69 | 46.92% | 46.92% | 46.92% | 3.08 pp | -8 | 6 | -1.33 |
| BTC Daily | transformer | Transformer | 130 | 58 | 72 | 44.62% | 44.62% | 44.62% | 5.38 pp | -14 | 6 | -2.33 |
| BTC Daily | rf | RandomForest | 130 | 54 | 76 | 41.54% | 41.54% | 41.54% | 8.46 pp | -22 | 6 | -3.67 |
| BTC Daily | xgb | XGBoost | 140 | 51 | 89 | 36.43% | 36.43% | 36.43% | 13.57 pp | -38 | 7 | -5.43 |
| BTC Daily | lstm | LSTM | 130 | 45 | 85 | 34.62% | 34.62% | 34.62% | 15.38 pp | -40 | 6 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 128 | 68 | 60 | 53.12% | 53.12% | 53.12% | 3.12 pp | 8 | 10 | 0.80 |
| BTC Market Hours | rf | RandomForest | 128 | 61 | 67 | 47.66% | 47.66% | 47.66% | 2.34 pp | -6 | 10 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 128 | 58 | 70 | 45.31% | 45.31% | 45.31% | 4.69 pp | -12 | 10 | -1.20 |
| BTC Market Hours | transformer | Transformer | 128 | 55 | 73 | 42.97% | 42.97% | 42.97% | 7.03 pp | -18 | 10 | -1.80 |
| BTC Market Hours | xgb | XGBoost | 128 | 52 | 76 | 40.62% | 40.62% | 40.62% | 9.38 pp | -24 | 10 | -2.40 |
| BTC Market Hours | lstm | LSTM | 128 | 49 | 79 | 38.28% | 38.28% | 38.28% | 11.72 pp | -30 | 10 | -3.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 127 | 61 | 66 | 48.03% | 48.03% | 48.03% | 1.97 pp | -5 | 11 | -0.45 |
| BTC Market Hours Daily | transformer | Transformer | 127 | 59 | 68 | 46.46% | 46.46% | 46.46% | 3.54 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | rf | RandomForest | 127 | 58 | 69 | 45.67% | 45.67% | 45.67% | 4.33 pp | -11 | 11 | -1.00 |
| BTC Market Hours Daily | nn | NN | 127 | 57 | 70 | 44.88% | 44.88% | 44.88% | 5.12 pp | -13 | 11 | -1.18 |
| BTC Market Hours Daily | xgb | XGBoost | 127 | 50 | 77 | 39.37% | 39.37% | 39.37% | 10.63 pp | -27 | 11 | -2.45 |
| BTC Market Hours Daily | lstm | LSTM | 127 | 47 | 80 | 37.01% | 37.01% | 37.01% | 12.99 pp | -33 | 11 | -3.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Hourly | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 105 | 55 | 50 | 52.38% | 52.38% | 52.38% | 2.38 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 105 | 54 | 51 | 51.43% | 51.43% | 51.43% | 1.43 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 105 | 51 | 54 | 48.57% | 48.57% | 48.57% | 1.43 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 105 | 49 | 56 | 46.67% | 46.67% | 46.67% | 3.33 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 105 | 47 | 58 | 44.76% | 44.76% | 44.76% | 5.24 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | nn | NN | 105 | 44 | 61 | 41.90% | 41.90% | 41.90% | 8.10 pp | -17 | 9 | -1.89 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 12 | 8 | 4 | 66.67% | 66.67% | 66.67% | 16.67 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 12 | 6 | 6 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | nn | NN | 12 | 5 | 7 | 41.67% | 41.67% | 41.67% | 8.33 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 12 | 4 | 8 | 33.33% | 33.33% | 33.33% | 16.67 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
