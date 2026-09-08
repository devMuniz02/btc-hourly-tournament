# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-08T14:35:48.876724+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 277 | 217 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 313 | 253 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 451 | 241 | 210 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-08 13:00:00+00:00 | 450 | 240 | 210 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 209 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 14:00:00+00:00 | 209 | 69 | 140 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 241 | 123 | 118 | 51.04% | 51.25% | 51.04% | 1.04 pp | 5 | 19 | 0.26 |
| BTC Market Hours Daily | transformer | Transformer | 240 | 118 | 122 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 14 | -0.21 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 240 | 116 | 124 | 48.33% | 48.33% | 48.33% | 1.67 pp | -8 | 20 | -0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 14 | -0.64 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 241 | 114 | 127 | 47.30% | 47.08% | 47.30% | 2.70 pp | -13 | 19 | -0.68 |
| BTC Market Hours Daily | nn | NN | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 20 | -0.70 |
| BTC Market Hours | transformer | Transformer | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 19 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 241 | 113 | 128 | 46.89% | 46.67% | 46.89% | 3.11 pp | -15 | 19 | -0.79 |
| Consolidated Market Hours | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Market Hours Daily | xgb | XGBoost | 69 | 32 | 37 | 46.38% | 46.38% | 46.38% | 3.62 pp | -5 | 6 | -0.83 |
| Consolidated Hourly | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 209 | 97 | 112 | 46.41% | 46.41% | 46.41% | 3.59 pp | -15 | 14 | -1.07 |
| BTC Market Hours | rf | RandomForest | 241 | 110 | 131 | 45.64% | 45.83% | 45.64% | 4.36 pp | -21 | 19 | -1.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 243 | 114 | 129 | 46.91% | 46.67% | 46.91% | 3.09 pp | -15 | 11 | -1.36 |
| BTC Market Hours Daily | xgb | XGBoost | 240 | 106 | 134 | 44.17% | 44.17% | 44.17% | 5.83 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 20 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 69 | 30 | 39 | 43.48% | 43.48% | 43.48% | 6.52 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 14 | -1.79 |
| Consolidated Market Hours | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 69 | 29 | 40 | 42.03% | 42.03% | 42.03% | 7.97 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 209 | 91 | 118 | 43.54% | 43.54% | 43.54% | 6.46 pp | -27 | 14 | -1.93 |
| BTC Market Hours | lstm | LSTM | 241 | 102 | 139 | 42.32% | 42.50% | 42.32% | 7.68 pp | -37 | 19 | -1.95 |
| BTC Daily | nn | NN | 243 | 110 | 133 | 45.27% | 45.00% | 45.27% | 4.73 pp | -23 | 11 | -2.09 |
| BTC Market Hours Daily | lstm | LSTM | 240 | 97 | 143 | 40.42% | 40.42% | 40.42% | 9.58 pp | -46 | 20 | -2.30 |
| BTC Hourly | transformer | Transformer | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 9 | -2.33 |
| Consolidated Hourly | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Daily/Hourly Refresh | nn | NN | 209 | 88 | 121 | 42.11% | 42.11% | 42.11% | 7.89 pp | -33 | 14 | -2.36 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 69 | 27 | 42 | 39.13% | 39.13% | 39.13% | 10.87 pp | -15 | 6 | -2.50 |
| Consolidated Market Hours | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 69 | 26 | 43 | 37.68% | 37.68% | 37.68% | 12.32 pp | -17 | 6 | -2.83 |
| BTC Hourly | nn | NN | 217 | 91 | 126 | 41.94% | 41.94% | 41.94% | 8.06 pp | -35 | 9 | -3.89 |
| BTC Daily | transformer | Transformer | 243 | 99 | 144 | 40.74% | 40.42% | 40.74% | 9.26 pp | -45 | 11 | -4.09 |
| BTC Hourly | rf | RandomForest | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 9 | -4.11 |
| BTC Daily | rf | RandomForest | 243 | 93 | 150 | 38.27% | 37.92% | 38.27% | 11.73 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 253 | 90 | 163 | 35.57% | 35.42% | 35.57% | 14.43 pp | -73 | 12 | -6.08 |
| BTC Hourly | lstm | LSTM | 217 | 80 | 137 | 36.87% | 36.87% | 36.87% | 13.13 pp | -57 | 9 | -6.33 |
| BTC Daily | lstm | LSTM | 243 | 82 | 161 | 33.74% | 34.17% | 33.74% | 16.26 pp | -79 | 11 | -7.18 |
| BTC Hourly | xgb | XGBoost | 217 | 75 | 142 | 34.56% | 34.56% | 34.56% | 15.44 pp | -67 | 9 | -7.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 217 | 106 | 111 | 48.85% | 48.85% | 48.85% | 1.15 pp | -5 | 9 | -0.56 |
| BTC Hourly | transformer | Transformer | 217 | 98 | 119 | 45.16% | 45.16% | 45.16% | 4.84 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 217 | 91 | 126 | 41.94% | 41.94% | 41.94% | 8.06 pp | -35 | 9 | -3.89 |
| BTC Hourly | rf | RandomForest | 217 | 90 | 127 | 41.47% | 41.47% | 41.47% | 8.53 pp | -37 | 9 | -4.11 |
| BTC Hourly | lstm | LSTM | 217 | 80 | 137 | 36.87% | 36.87% | 36.87% | 13.13 pp | -57 | 9 | -6.33 |
| BTC Hourly | xgb | XGBoost | 217 | 75 | 142 | 34.56% | 34.56% | 34.56% | 15.44 pp | -67 | 9 | -7.44 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 243 | 114 | 129 | 46.91% | 46.67% | 46.91% | 3.09 pp | -15 | 11 | -1.36 |
| BTC Daily | nn | NN | 243 | 110 | 133 | 45.27% | 45.00% | 45.27% | 4.73 pp | -23 | 11 | -2.09 |
| BTC Daily | transformer | Transformer | 243 | 99 | 144 | 40.74% | 40.42% | 40.74% | 9.26 pp | -45 | 11 | -4.09 |
| BTC Daily | rf | RandomForest | 243 | 93 | 150 | 38.27% | 37.92% | 38.27% | 11.73 pp | -57 | 11 | -5.18 |
| BTC Daily | xgb | XGBoost | 253 | 90 | 163 | 35.57% | 35.42% | 35.57% | 14.43 pp | -73 | 12 | -6.08 |
| BTC Daily | lstm | LSTM | 243 | 82 | 161 | 33.74% | 34.17% | 33.74% | 16.26 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 241 | 123 | 118 | 51.04% | 51.25% | 51.04% | 1.04 pp | 5 | 19 | 0.26 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 241 | 114 | 127 | 47.30% | 47.08% | 47.30% | 2.70 pp | -13 | 19 | -0.68 |
| BTC Market Hours | transformer | Transformer | 241 | 113 | 128 | 46.89% | 47.08% | 46.89% | 3.11 pp | -15 | 19 | -0.79 |
| BTC Market Hours | xgb | XGBoost | 241 | 113 | 128 | 46.89% | 46.67% | 46.89% | 3.11 pp | -15 | 19 | -0.79 |
| BTC Market Hours | rf | RandomForest | 241 | 110 | 131 | 45.64% | 45.83% | 45.64% | 4.36 pp | -21 | 19 | -1.11 |
| BTC Market Hours | lstm | LSTM | 241 | 102 | 139 | 42.32% | 42.50% | 42.32% | 7.68 pp | -37 | 19 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 240 | 118 | 122 | 49.17% | 49.17% | 49.17% | 0.83 pp | -4 | 20 | -0.20 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 240 | 116 | 124 | 48.33% | 48.33% | 48.33% | 1.67 pp | -8 | 20 | -0.40 |
| BTC Market Hours Daily | nn | NN | 240 | 113 | 127 | 47.08% | 47.08% | 47.08% | 2.92 pp | -14 | 20 | -0.70 |
| BTC Market Hours Daily | xgb | XGBoost | 240 | 106 | 134 | 44.17% | 44.17% | 44.17% | 5.83 pp | -28 | 20 | -1.40 |
| BTC Market Hours Daily | rf | RandomForest | 240 | 105 | 135 | 43.75% | 43.75% | 43.75% | 6.25 pp | -30 | 20 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 240 | 97 | 143 | 40.42% | 40.42% | 40.42% | 9.58 pp | -46 | 20 | -2.30 |

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
