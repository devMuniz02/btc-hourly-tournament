# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T03:01:08.985738+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 220 | 160 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 256 | 196 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 353 | 184 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 353 | 184 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 155 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 00:00:00+00:00 | 155 | 40 | 115 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 7 | 0.86 |
| BTC Market Hours Daily | transformer | Transformer | 184 | 96 | 88 | 52.17% | 52.17% | 52.17% | 2.17 pp | 8 | 16 | 0.50 |
| BTC Market Hours | nn | NN | 184 | 94 | 90 | 51.09% | 51.09% | 51.09% | 1.09 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | transformer | Transformer | 184 | 91 | 93 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 184 | 89 | 95 | 48.37% | 48.37% | 48.37% | 1.63 pp | -6 | 16 | -0.38 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 16 | -0.75 |
| BTC Hourly | transformer | Transformer | 160 | 77 | 83 | 48.12% | 48.12% | 48.12% | 1.87 pp | -6 | 7 | -0.86 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 9 | -1.33 |
| BTC Market Hours Daily | rf | RandomForest | 184 | 81 | 103 | 44.02% | 44.02% | 44.02% | 5.98 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| BTC Market Hours | xgb | XGBoost | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 15 | -1.87 |
| BTC Market Hours Daily | xgb | XGBoost | 184 | 77 | 107 | 41.85% | 41.85% | 41.85% | 8.15 pp | -30 | 16 | -1.88 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| BTC Daily | nn | NN | 186 | 83 | 103 | 44.62% | 44.62% | 44.62% | 5.38 pp | -20 | 9 | -2.22 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 184 | 74 | 110 | 40.22% | 40.22% | 40.22% | 9.78 pp | -36 | 16 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| BTC Daily | transformer | Transformer | 186 | 81 | 105 | 43.55% | 43.55% | 43.55% | 6.45 pp | -24 | 9 | -2.67 |
| BTC Hourly | nn | NN | 160 | 68 | 92 | 42.50% | 42.50% | 42.50% | 7.50 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 7 | -3.71 |
| BTC Daily | rf | RandomForest | 186 | 74 | 112 | 39.78% | 39.78% | 39.78% | 10.22 pp | -38 | 9 | -4.22 |
| BTC Daily | xgb | XGBoost | 196 | 72 | 124 | 36.73% | 36.73% | 36.73% | 13.27 pp | -52 | 10 | -5.20 |
| BTC Daily | lstm | LSTM | 186 | 65 | 121 | 34.95% | 34.95% | 34.95% | 15.05 pp | -56 | 9 | -6.22 |
| BTC Hourly | lstm | LSTM | 160 | 58 | 102 | 36.25% | 36.25% | 36.25% | 13.75 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 160 | 56 | 104 | 35.00% | 35.00% | 35.00% | 15.00 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 160 | 83 | 77 | 51.88% | 51.88% | 51.88% | 1.88 pp | 6 | 7 | 0.86 |
| BTC Hourly | transformer | Transformer | 160 | 77 | 83 | 48.12% | 48.12% | 48.12% | 1.87 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 160 | 68 | 92 | 42.50% | 42.50% | 42.50% | 7.50 pp | -24 | 7 | -3.43 |
| BTC Hourly | rf | RandomForest | 160 | 67 | 93 | 41.88% | 41.88% | 41.88% | 8.12 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 160 | 58 | 102 | 36.25% | 36.25% | 36.25% | 13.75 pp | -44 | 7 | -6.29 |
| BTC Hourly | xgb | XGBoost | 160 | 56 | 104 | 35.00% | 35.00% | 35.00% | 15.00 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 186 | 87 | 99 | 46.77% | 46.77% | 46.77% | 3.23 pp | -12 | 9 | -1.33 |
| BTC Daily | nn | NN | 186 | 83 | 103 | 44.62% | 44.62% | 44.62% | 5.38 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 186 | 81 | 105 | 43.55% | 43.55% | 43.55% | 6.45 pp | -24 | 9 | -2.67 |
| BTC Daily | rf | RandomForest | 186 | 74 | 112 | 39.78% | 39.78% | 39.78% | 10.22 pp | -38 | 9 | -4.22 |
| BTC Daily | xgb | XGBoost | 196 | 72 | 124 | 36.73% | 36.73% | 36.73% | 13.27 pp | -52 | 10 | -5.20 |
| BTC Daily | lstm | LSTM | 186 | 65 | 121 | 34.95% | 34.95% | 34.95% | 15.05 pp | -56 | 9 | -6.22 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 184 | 94 | 90 | 51.09% | 51.09% | 51.09% | 1.09 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 184 | 91 | 93 | 49.46% | 49.46% | 49.46% | 0.54 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 184 | 85 | 99 | 46.20% | 46.20% | 46.20% | 3.80 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 184 | 79 | 105 | 42.93% | 42.93% | 42.93% | 7.07 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 184 | 78 | 106 | 42.39% | 42.39% | 42.39% | 7.61 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 184 | 96 | 88 | 52.17% | 52.17% | 52.17% | 2.17 pp | 8 | 16 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 184 | 89 | 95 | 48.37% | 48.37% | 48.37% | 1.63 pp | -6 | 16 | -0.38 |
| BTC Market Hours Daily | nn | NN | 184 | 86 | 98 | 46.74% | 46.74% | 46.74% | 3.26 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 184 | 81 | 103 | 44.02% | 44.02% | 44.02% | 5.98 pp | -22 | 16 | -1.38 |
| BTC Market Hours Daily | xgb | XGBoost | 184 | 77 | 107 | 41.85% | 41.85% | 41.85% | 8.15 pp | -30 | 16 | -1.88 |
| BTC Market Hours Daily | lstm | LSTM | 184 | 74 | 110 | 40.22% | 40.22% | 40.22% | 9.78 pp | -36 | 16 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Hourly | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 155 | 79 | 76 | 50.97% | 50.97% | 50.97% | 0.97 pp | 3 | 12 | 0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 155 | 76 | 79 | 49.03% | 49.03% | 49.03% | 0.97 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 155 | 72 | 83 | 46.45% | 46.45% | 46.45% | 3.55 pp | -11 | 12 | -0.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 155 | 69 | 86 | 44.52% | 44.52% | 44.52% | 5.48 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 155 | 68 | 87 | 43.87% | 43.87% | 43.87% | 6.13 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | nn | NN | 155 | 64 | 91 | 41.29% | 41.29% | 41.29% | 8.71 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 40 | 20 | 20 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 40 | 18 | 22 | 45.00% | 45.00% | 45.00% | 5.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 40 | 16 | 24 | 40.00% | 40.00% | 40.00% | 10.00 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 40 | 15 | 25 | 37.50% | 37.50% | 37.50% | 12.50 pp | -10 | 4 | -2.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
