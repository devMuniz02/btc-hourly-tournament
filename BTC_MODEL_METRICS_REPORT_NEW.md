# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T16:13:20.377304+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 278 | 218 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 314 | 254 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 15:00:00+00:00 | 454 | 242 | 212 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 15:00:00+00:00 | 454 | 242 | 212 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 242 | 124 | 118 | 51.24% | 51.67% | 51.24% | 1.24 pp | 6 | 19 | 0.32 |
| BTC Market Hours Daily | transformer | Transformer | 242 | 119 | 123 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 48.75% | 48.76% | 1.24 pp | -6 | 20 | -0.30 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 218 | 107 | 111 | 49.08% | 49.08% | 49.08% | 0.92 pp | -4 | 10 | -0.40 |
| BTC Market Hours Daily | nn | NN | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 20 | -0.60 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 19 | -0.63 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| BTC Market Hours | transformer | Transformer | 242 | 113 | 129 | 46.69% | 47.08% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 242 | 110 | 132 | 45.45% | 45.83% | 45.45% | 4.55 pp | -22 | 19 | -1.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 244 | 115 | 129 | 47.13% | 47.08% | 47.13% | 2.87 pp | -14 | 11 | -1.27 |
| BTC Market Hours Daily | xgb | XGBoost | 242 | 108 | 134 | 44.63% | 44.17% | 44.63% | 5.37 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 242 | 107 | 135 | 44.21% | 43.75% | 44.21% | 5.79 pp | -28 | 20 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| BTC Market Hours | lstm | LSTM | 242 | 103 | 139 | 42.56% | 42.92% | 42.56% | 7.44 pp | -36 | 19 | -1.89 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| BTC Daily | nn | NN | 244 | 110 | 134 | 45.08% | 44.58% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Hourly | transformer | Transformer | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 242 | 98 | 144 | 40.50% | 40.83% | 40.50% | 9.50 pp | -46 | 20 | -2.30 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 218 | 92 | 126 | 42.20% | 42.20% | 42.20% | 7.80 pp | -34 | 10 | -3.40 |
| BTC Hourly | rf | RandomForest | 218 | 91 | 127 | 41.74% | 41.74% | 41.74% | 8.26 pp | -36 | 10 | -3.60 |
| BTC Daily | transformer | Transformer | 244 | 99 | 145 | 40.57% | 40.00% | 40.57% | 9.43 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 244 | 93 | 151 | 38.11% | 37.50% | 38.11% | 11.89 pp | -58 | 11 | -5.27 |
| BTC Hourly | lstm | LSTM | 218 | 81 | 137 | 37.16% | 37.16% | 37.16% | 12.84 pp | -56 | 10 | -5.60 |
| BTC Daily | xgb | XGBoost | 254 | 90 | 164 | 35.43% | 35.42% | 35.43% | 14.57 pp | -74 | 12 | -6.17 |
| BTC Hourly | xgb | XGBoost | 218 | 75 | 143 | 34.40% | 34.40% | 34.40% | 15.60 pp | -68 | 10 | -6.80 |
| BTC Daily | lstm | LSTM | 244 | 82 | 162 | 33.61% | 33.75% | 33.61% | 16.39 pp | -80 | 11 | -7.27 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 218 | 107 | 111 | 49.08% | 49.08% | 49.08% | 0.92 pp | -4 | 10 | -0.40 |
| BTC Hourly | transformer | Transformer | 218 | 98 | 120 | 44.95% | 44.95% | 44.95% | 5.05 pp | -22 | 10 | -2.20 |
| BTC Hourly | nn | NN | 218 | 92 | 126 | 42.20% | 42.20% | 42.20% | 7.80 pp | -34 | 10 | -3.40 |
| BTC Hourly | rf | RandomForest | 218 | 91 | 127 | 41.74% | 41.74% | 41.74% | 8.26 pp | -36 | 10 | -3.60 |
| BTC Hourly | lstm | LSTM | 218 | 81 | 137 | 37.16% | 37.16% | 37.16% | 12.84 pp | -56 | 10 | -5.60 |
| BTC Hourly | xgb | XGBoost | 218 | 75 | 143 | 34.40% | 34.40% | 34.40% | 15.60 pp | -68 | 10 | -6.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 244 | 115 | 129 | 47.13% | 47.08% | 47.13% | 2.87 pp | -14 | 11 | -1.27 |
| BTC Daily | nn | NN | 244 | 110 | 134 | 45.08% | 44.58% | 45.08% | 4.92 pp | -24 | 11 | -2.18 |
| BTC Daily | transformer | Transformer | 244 | 99 | 145 | 40.57% | 40.00% | 40.57% | 9.43 pp | -46 | 11 | -4.18 |
| BTC Daily | rf | RandomForest | 244 | 93 | 151 | 38.11% | 37.50% | 38.11% | 11.89 pp | -58 | 11 | -5.27 |
| BTC Daily | xgb | XGBoost | 254 | 90 | 164 | 35.43% | 35.42% | 35.43% | 14.57 pp | -74 | 12 | -6.17 |
| BTC Daily | lstm | LSTM | 244 | 82 | 162 | 33.61% | 33.75% | 33.61% | 16.39 pp | -80 | 11 | -7.27 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 242 | 124 | 118 | 51.24% | 51.67% | 51.24% | 1.24 pp | 6 | 19 | 0.32 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 19 | -0.63 |
| BTC Market Hours | transformer | Transformer | 242 | 113 | 129 | 46.69% | 47.08% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | xgb | XGBoost | 242 | 113 | 129 | 46.69% | 46.67% | 46.69% | 3.31 pp | -16 | 19 | -0.84 |
| BTC Market Hours | rf | RandomForest | 242 | 110 | 132 | 45.45% | 45.83% | 45.45% | 4.55 pp | -22 | 19 | -1.16 |
| BTC Market Hours | lstm | LSTM | 242 | 103 | 139 | 42.56% | 42.92% | 42.56% | 7.44 pp | -36 | 19 | -1.89 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 242 | 119 | 123 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 242 | 118 | 124 | 48.76% | 48.75% | 48.76% | 1.24 pp | -6 | 20 | -0.30 |
| BTC Market Hours Daily | nn | NN | 242 | 115 | 127 | 47.52% | 47.50% | 47.52% | 2.48 pp | -12 | 20 | -0.60 |
| BTC Market Hours Daily | xgb | XGBoost | 242 | 108 | 134 | 44.63% | 44.17% | 44.63% | 5.37 pp | -26 | 20 | -1.30 |
| BTC Market Hours Daily | rf | RandomForest | 242 | 107 | 135 | 44.21% | 43.75% | 44.21% | 5.79 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | lstm | LSTM | 242 | 98 | 144 | 40.50% | 40.83% | 40.50% | 9.50 pp | -46 | 20 | -2.30 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
