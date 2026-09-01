# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-01T04:53:54.700703+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 157 | 97 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 193 | 133 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 238 | 121 | 117 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 237 | 120 | 117 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 99 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 19:00:00+00:00 | 99 | 9 | 90 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| BTC Market Hours | nn | NN | 121 | 64 | 57 | 52.89% | 52.89% | 52.89% | 2.89 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 6 | 0.17 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| BTC Market Hours | rf | RandomForest | 121 | 59 | 62 | 48.76% | 48.76% | 48.76% | 1.24 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Hourly | nn | NN | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 10 | -0.90 |
| BTC Market Hours Daily | rf | RandomForest | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | nn | NN | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |
| BTC Daily | transformer | Transformer | 123 | 57 | 66 | 46.34% | 46.34% | 46.34% | 3.66 pp | -9 | 6 | -1.50 |
| BTC Market Hours | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 97 | 45 | 52 | 46.39% | 46.39% | 46.39% | 3.61 pp | -7 | 4 | -1.75 |
| BTC Daily | nn | NN | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 6 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 11 | -2.36 |
| BTC Market Hours | lstm | LSTM | 121 | 47 | 74 | 38.84% | 38.84% | 38.84% | 11.16 pp | -27 | 10 | -2.70 |
| BTC Market Hours Daily | lstm | LSTM | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 11 | -2.73 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Daily | rf | RandomForest | 123 | 51 | 72 | 41.46% | 41.46% | 41.46% | 8.54 pp | -21 | 6 | -3.50 |
| BTC Hourly | rf | RandomForest | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 4 | -4.75 |
| BTC Daily | xgb | XGBoost | 133 | 48 | 85 | 36.09% | 36.09% | 36.09% | 13.91 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 123 | 42 | 81 | 34.15% | 34.15% | 34.15% | 15.85 pp | -39 | 6 | -6.50 |
| BTC Hourly | xgb | XGBoost | 97 | 32 | 65 | 32.99% | 32.99% | 32.99% | 17.01 pp | -33 | 4 | -8.25 |
| BTC Hourly | lstm | LSTM | 97 | 31 | 66 | 31.96% | 31.96% | 31.96% | 18.04 pp | -35 | 4 | -8.75 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 4 | -0.75 |
| BTC Hourly | transformer | Transformer | 97 | 47 | 50 | 48.45% | 48.45% | 48.45% | 1.55 pp | -3 | 4 | -0.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 97 | 45 | 52 | 46.39% | 46.39% | 46.39% | 3.61 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 97 | 39 | 58 | 40.21% | 40.21% | 40.21% | 9.79 pp | -19 | 4 | -4.75 |
| BTC Hourly | xgb | XGBoost | 97 | 32 | 65 | 32.99% | 32.99% | 32.99% | 17.01 pp | -33 | 4 | -8.25 |
| BTC Hourly | lstm | LSTM | 97 | 31 | 66 | 31.96% | 31.96% | 31.96% | 18.04 pp | -35 | 4 | -8.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 123 | 62 | 61 | 50.41% | 50.41% | 50.41% | 0.41 pp | 1 | 6 | 0.17 |
| BTC Daily | transformer | Transformer | 123 | 57 | 66 | 46.34% | 46.34% | 46.34% | 3.66 pp | -9 | 6 | -1.50 |
| BTC Daily | nn | NN | 123 | 56 | 67 | 45.53% | 45.53% | 45.53% | 4.47 pp | -11 | 6 | -1.83 |
| BTC Daily | rf | RandomForest | 123 | 51 | 72 | 41.46% | 41.46% | 41.46% | 8.54 pp | -21 | 6 | -3.50 |
| BTC Daily | xgb | XGBoost | 133 | 48 | 85 | 36.09% | 36.09% | 36.09% | 13.91 pp | -37 | 7 | -5.29 |
| BTC Daily | lstm | LSTM | 123 | 42 | 81 | 34.15% | 34.15% | 34.15% | 15.85 pp | -39 | 6 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 121 | 64 | 57 | 52.89% | 52.89% | 52.89% | 2.89 pp | 7 | 10 | 0.70 |
| BTC Market Hours | rf | RandomForest | 121 | 59 | 62 | 48.76% | 48.76% | 48.76% | 1.24 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 121 | 56 | 65 | 46.28% | 46.28% | 46.28% | 3.72 pp | -9 | 10 | -0.90 |
| BTC Market Hours | transformer | Transformer | 121 | 52 | 69 | 42.98% | 42.98% | 42.98% | 7.02 pp | -17 | 10 | -1.70 |
| BTC Market Hours | xgb | XGBoost | 121 | 49 | 72 | 40.50% | 40.50% | 40.50% | 9.50 pp | -23 | 10 | -2.30 |
| BTC Market Hours | lstm | LSTM | 121 | 47 | 74 | 38.84% | 38.84% | 38.84% | 11.16 pp | -27 | 10 | -2.70 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 120 | 60 | 60 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 11 | 0.00 |
| BTC Market Hours Daily | rf | RandomForest | 120 | 55 | 65 | 45.83% | 45.83% | 45.83% | 4.17 pp | -10 | 11 | -0.91 |
| BTC Market Hours Daily | nn | NN | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | transformer | Transformer | 120 | 54 | 66 | 45.00% | 45.00% | 45.00% | 5.00 pp | -12 | 11 | -1.09 |
| BTC Market Hours Daily | xgb | XGBoost | 120 | 47 | 73 | 39.17% | 39.17% | 39.17% | 10.83 pp | -26 | 11 | -2.36 |
| BTC Market Hours Daily | lstm | LSTM | 120 | 45 | 75 | 37.50% | 37.50% | 37.50% | 12.50 pp | -30 | 11 | -2.73 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 53 | 46 | 53.54% | 53.54% | 53.54% | 3.54 pp | 7 | 9 | 0.78 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 43 | 56 | 43.43% | 43.43% | 43.43% | 6.57 pp | -13 | 9 | -1.44 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
