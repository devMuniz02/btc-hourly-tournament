# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T21:02:18.575230+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 151 | 91 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 187 | 127 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 20:00:00+00:00 | 228 | 115 | 113 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 20:00:00+00:00 | 228 | 115 | 113 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T16:00:00+00:00 | 93 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T16:00:00+00:00 | 93 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T16:00:00+00:00 | 93 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T16:00:00+00:00 | 94 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 115 | 62 | 53 | 53.91% | 53.91% | 53.91% | 3.91 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 115 | 57 | 58 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| Consolidated Hourly | nn | NN | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 4 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 117 | 57 | 60 | 48.72% | 48.72% | 48.72% | 1.28 pp | -3 | 6 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| BTC Market Hours | rf | RandomForest | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 115 | 53 | 62 | 46.09% | 46.09% | 46.09% | 3.91 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | nn | NN | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 6 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 9 | -1.22 |
| BTC Hourly | nn | NN | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 4 | -1.75 |
| BTC Daily | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| BTC Market Hours | transformer | Transformer | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | xgb | XGBoost | 115 | 45 | 70 | 39.13% | 39.13% | 39.13% | 10.87 pp | -25 | 10 | -2.50 |
| BTC Market Hours | xgb | XGBoost | 115 | 46 | 69 | 40.00% | 40.00% | 40.00% | 10.00 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 115 | 45 | 70 | 39.13% | 39.13% | 39.13% | 10.87 pp | -25 | 9 | -2.78 |
| Consolidated Market Hours Daily | lstm | LSTM | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 115 | 42 | 73 | 36.52% | 36.52% | 36.52% | 13.48 pp | -31 | 10 | -3.10 |
| BTC Daily | rf | RandomForest | 117 | 47 | 70 | 40.17% | 40.17% | 40.17% | 9.83 pp | -23 | 6 | -3.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |
| BTC Hourly | rf | RandomForest | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 4 | -4.25 |
| BTC Daily | xgb | XGBoost | 127 | 46 | 81 | 36.22% | 36.22% | 36.22% | 13.78 pp | -35 | 7 | -5.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |
| BTC Daily | lstm | LSTM | 117 | 41 | 76 | 35.04% | 35.04% | 35.04% | 14.96 pp | -35 | 6 | -5.83 |
| BTC Hourly | lstm | LSTM | 91 | 30 | 61 | 32.97% | 32.97% | 32.97% | 17.03 pp | -31 | 4 | -7.75 |
| BTC Hourly | xgb | XGBoost | 91 | 30 | 61 | 32.97% | 32.97% | 32.97% | 17.03 pp | -31 | 4 | -7.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 91 | 45 | 46 | 49.45% | 49.45% | 49.45% | 0.55 pp | -1 | 4 | -0.25 |
| BTC Hourly | nn | NN | 91 | 43 | 48 | 47.25% | 47.25% | 47.25% | 2.75 pp | -5 | 4 | -1.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 91 | 42 | 49 | 46.15% | 46.15% | 46.15% | 3.85 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 91 | 37 | 54 | 40.66% | 40.66% | 40.66% | 9.34 pp | -17 | 4 | -4.25 |
| BTC Hourly | lstm | LSTM | 91 | 30 | 61 | 32.97% | 32.97% | 32.97% | 17.03 pp | -31 | 4 | -7.75 |
| BTC Hourly | xgb | XGBoost | 91 | 30 | 61 | 32.97% | 32.97% | 32.97% | 17.03 pp | -31 | 4 | -7.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 117 | 57 | 60 | 48.72% | 48.72% | 48.72% | 1.28 pp | -3 | 6 | -0.50 |
| BTC Daily | nn | NN | 117 | 55 | 62 | 47.01% | 47.01% | 47.01% | 2.99 pp | -7 | 6 | -1.17 |
| BTC Daily | transformer | Transformer | 117 | 53 | 64 | 45.30% | 45.30% | 45.30% | 4.70 pp | -11 | 6 | -1.83 |
| BTC Daily | rf | RandomForest | 117 | 47 | 70 | 40.17% | 40.17% | 40.17% | 9.83 pp | -23 | 6 | -3.83 |
| BTC Daily | xgb | XGBoost | 127 | 46 | 81 | 36.22% | 36.22% | 36.22% | 13.78 pp | -35 | 7 | -5.00 |
| BTC Daily | lstm | LSTM | 117 | 41 | 76 | 35.04% | 35.04% | 35.04% | 14.96 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 115 | 62 | 53 | 53.91% | 53.91% | 53.91% | 3.91 pp | 9 | 9 | 1.00 |
| BTC Market Hours | rf | RandomForest | 115 | 54 | 61 | 46.96% | 46.96% | 46.96% | 3.04 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 115 | 53 | 62 | 46.09% | 46.09% | 46.09% | 3.91 pp | -9 | 9 | -1.00 |
| BTC Market Hours | transformer | Transformer | 115 | 48 | 67 | 41.74% | 41.74% | 41.74% | 8.26 pp | -19 | 9 | -2.11 |
| BTC Market Hours | xgb | XGBoost | 115 | 46 | 69 | 40.00% | 40.00% | 40.00% | 10.00 pp | -23 | 9 | -2.56 |
| BTC Market Hours | lstm | LSTM | 115 | 45 | 70 | 39.13% | 39.13% | 39.13% | 10.87 pp | -25 | 9 | -2.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 115 | 57 | 58 | 49.57% | 49.57% | 49.57% | 0.43 pp | -1 | 10 | -0.10 |
| BTC Market Hours Daily | nn | NN | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | transformer | Transformer | 115 | 52 | 63 | 45.22% | 45.22% | 45.22% | 4.78 pp | -11 | 10 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 115 | 51 | 64 | 44.35% | 44.35% | 44.35% | 5.65 pp | -13 | 10 | -1.30 |
| BTC Market Hours Daily | xgb | XGBoost | 115 | 45 | 70 | 39.13% | 39.13% | 39.13% | 10.87 pp | -25 | 10 | -2.50 |
| BTC Market Hours Daily | lstm | LSTM | 115 | 42 | 73 | 36.52% | 36.52% | 36.52% | 13.48 pp | -31 | 10 | -3.10 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Hourly | lstm | LSTM | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 93 | 48 | 45 | 51.61% | 51.61% | 51.61% | 1.61 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 93 | 47 | 46 | 50.54% | 50.54% | 50.54% | 0.54 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 93 | 46 | 47 | 49.46% | 49.46% | 49.46% | 0.54 pp | -1 | 9 | -0.11 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 93 | 44 | 49 | 47.31% | 47.31% | 47.31% | 2.69 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 93 | 41 | 52 | 44.09% | 44.09% | 44.09% | 5.91 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | transformer | Transformer | 6 | 5 | 1 | 83.33% | 83.33% | 83.33% | 33.33 pp | 4 | 1 | 4.00 |
| Consolidated Market Hours | rf | RandomForest | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 6 | 4 | 2 | 66.67% | 66.67% | 66.67% | 16.67 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 6 | 2 | 4 | 33.33% | 33.33% | 33.33% | 16.67 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 6 | 1 | 5 | 16.67% | 16.67% | 16.67% | 33.33 pp | -4 | 1 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 7 | 6 | 1 | 85.71% | 85.71% | 85.71% | 35.71 pp | 5 | 1 | 5.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 7 | 4 | 3 | 57.14% | 57.14% | 57.14% | 7.14 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 7 | 2 | 5 | 28.57% | 28.57% | 28.57% | 21.43 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 7 | 1 | 6 | 14.29% | 14.29% | 14.29% | 35.71 pp | -5 | 1 | -5.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
