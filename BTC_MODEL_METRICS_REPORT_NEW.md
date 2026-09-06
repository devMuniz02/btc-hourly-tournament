# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T00:19:21.998346+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 234 | 174 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 270 | 210 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 23:00:00+00:00 | 379 | 198 | 181 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 23:00:00+00:00 | 379 | 198 | 181 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 171 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 19:00:00+00:00 | 171 | 48 | 123 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 198 | 103 | 95 | 52.02% | 52.02% | 52.02% | 2.02 pp | 8 | 17 | 0.47 |
| BTC Market Hours | nn | NN | 198 | 101 | 97 | 51.01% | 51.01% | 51.01% | 1.01 pp | 4 | 16 | 0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 174 | 88 | 86 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 198 | 97 | 101 | 48.99% | 48.99% | 48.99% | 1.01 pp | -4 | 16 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 17 | -0.47 |
| BTC Market Hours Daily | nn | NN | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 17 | -0.47 |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 198 | 93 | 105 | 46.97% | 46.97% | 46.97% | 3.03 pp | -12 | 16 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 198 | 90 | 108 | 45.45% | 45.45% | 45.45% | 4.55 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 198 | 87 | 111 | 43.94% | 43.94% | 43.94% | 6.06 pp | -24 | 17 | -1.41 |
| BTC Hourly | transformer | Transformer | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 8 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 198 | 86 | 112 | 43.43% | 43.43% | 43.43% | 6.57 pp | -26 | 16 | -1.62 |
| BTC Market Hours | lstm | LSTM | 198 | 84 | 114 | 42.42% | 42.42% | 42.42% | 7.58 pp | -30 | 16 | -1.88 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 9 | -2.00 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 198 | 82 | 116 | 41.41% | 41.41% | 41.41% | 8.59 pp | -34 | 17 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 198 | 81 | 117 | 40.91% | 40.91% | 40.91% | 9.09 pp | -36 | 17 | -2.12 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| BTC Hourly | rf | RandomForest | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Hourly | nn | NN | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| BTC Daily | rf | RandomForest | 200 | 77 | 123 | 38.50% | 38.50% | 38.50% | 11.50 pp | -46 | 9 | -5.11 |
| BTC Hourly | lstm | LSTM | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 8 | -6.00 |
| BTC Daily | xgb | XGBoost | 210 | 74 | 136 | 35.24% | 35.24% | 35.24% | 14.76 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 200 | 67 | 133 | 33.50% | 33.50% | 33.50% | 16.50 pp | -66 | 9 | -7.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 174 | 88 | 86 | 50.57% | 50.57% | 50.57% | 0.57 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 174 | 81 | 93 | 46.55% | 46.55% | 46.55% | 3.45 pp | -12 | 8 | -1.50 |
| BTC Hourly | rf | RandomForest | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 8 | -3.25 |
| BTC Hourly | nn | NN | 174 | 73 | 101 | 41.95% | 41.95% | 41.95% | 8.05 pp | -28 | 8 | -3.50 |
| BTC Hourly | lstm | LSTM | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 8 | -6.00 |
| BTC Hourly | xgb | XGBoost | 174 | 63 | 111 | 36.21% | 36.21% | 36.21% | 13.79 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 200 | 96 | 104 | 48.00% | 48.00% | 48.00% | 2.00 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 200 | 91 | 109 | 45.50% | 45.50% | 45.50% | 4.50 pp | -18 | 9 | -2.00 |
| BTC Daily | transformer | Transformer | 200 | 85 | 115 | 42.50% | 42.50% | 42.50% | 7.50 pp | -30 | 9 | -3.33 |
| BTC Daily | rf | RandomForest | 200 | 77 | 123 | 38.50% | 38.50% | 38.50% | 11.50 pp | -46 | 9 | -5.11 |
| BTC Daily | xgb | XGBoost | 210 | 74 | 136 | 35.24% | 35.24% | 35.24% | 14.76 pp | -62 | 10 | -6.20 |
| BTC Daily | lstm | LSTM | 200 | 67 | 133 | 33.50% | 33.50% | 33.50% | 16.50 pp | -66 | 9 | -7.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 198 | 101 | 97 | 51.01% | 51.01% | 51.01% | 1.01 pp | 4 | 16 | 0.25 |
| BTC Market Hours | transformer | Transformer | 198 | 97 | 101 | 48.99% | 48.99% | 48.99% | 1.01 pp | -4 | 16 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 198 | 93 | 105 | 46.97% | 46.97% | 46.97% | 3.03 pp | -12 | 16 | -0.75 |
| BTC Market Hours | rf | RandomForest | 198 | 90 | 108 | 45.45% | 45.45% | 45.45% | 4.55 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 198 | 86 | 112 | 43.43% | 43.43% | 43.43% | 6.57 pp | -26 | 16 | -1.62 |
| BTC Market Hours | lstm | LSTM | 198 | 84 | 114 | 42.42% | 42.42% | 42.42% | 7.58 pp | -30 | 16 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 198 | 103 | 95 | 52.02% | 52.02% | 52.02% | 2.02 pp | 8 | 17 | 0.47 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 17 | -0.47 |
| BTC Market Hours Daily | nn | NN | 198 | 95 | 103 | 47.98% | 47.98% | 47.98% | 2.02 pp | -8 | 17 | -0.47 |
| BTC Market Hours Daily | rf | RandomForest | 198 | 87 | 111 | 43.94% | 43.94% | 43.94% | 6.06 pp | -24 | 17 | -1.41 |
| BTC Market Hours Daily | xgb | XGBoost | 198 | 82 | 116 | 41.41% | 41.41% | 41.41% | 8.59 pp | -34 | 17 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 198 | 81 | 117 | 40.91% | 40.91% | 40.91% | 9.09 pp | -36 | 17 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 171 | 85 | 86 | 49.71% | 49.71% | 49.71% | 0.29 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 171 | 83 | 88 | 48.54% | 48.54% | 48.54% | 1.46 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 171 | 78 | 93 | 45.61% | 45.61% | 45.61% | 4.39 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 171 | 72 | 99 | 42.11% | 42.11% | 42.11% | 7.89 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 18 | 30 | 37.50% | 37.50% | 37.50% | 12.50 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
