# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-09T09:52:12.668344+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 289 | 229 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 325 | 265 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 474 | 253 | 221 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-09 00:00:00+00:00 | 474 | 253 | 221 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 221 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 221 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 75 | 146 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-31 20:00:00+00:00 | 221 | 75 | 146 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 253 | 132 | 121 | 52.17% | 52.08% | 52.17% | 2.17 pp | 11 | 20 | 0.55 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 229 | 115 | 114 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Market Hours Daily | transformer | Transformer | 253 | 126 | 127 | 49.80% | 50.00% | 49.80% | 0.20 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 253 | 125 | 128 | 49.41% | 48.75% | 49.41% | 0.59 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | nn | NN | 253 | 122 | 131 | 48.22% | 47.92% | 48.22% | 1.78 pp | -9 | 21 | -0.43 |
| Consolidated Hourly | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| BTC Market Hours | transformer | Transformer | 253 | 121 | 132 | 47.83% | 48.33% | 47.83% | 2.17 pp | -11 | 20 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 253 | 118 | 135 | 46.64% | 46.25% | 46.64% | 3.36 pp | -17 | 20 | -0.85 |
| BTC Market Hours | xgb | XGBoost | 253 | 118 | 135 | 46.64% | 45.83% | 46.64% | 3.36 pp | -17 | 20 | -0.85 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| BTC Market Hours | rf | RandomForest | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 20 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 253 | 112 | 141 | 44.27% | 44.58% | 44.27% | 5.73 pp | -29 | 21 | -1.38 |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 255 | 119 | 136 | 46.67% | 46.67% | 46.67% | 3.33 pp | -17 | 11 | -1.55 |
| BTC Market Hours Daily | rf | RandomForest | 253 | 110 | 143 | 43.48% | 42.92% | 43.48% | 6.52 pp | -33 | 21 | -1.57 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Hourly | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| BTC Daily | nn | NN | 255 | 116 | 139 | 45.49% | 44.17% | 45.49% | 4.51 pp | -23 | 11 | -2.09 |
| BTC Market Hours | lstm | LSTM | 253 | 105 | 148 | 41.50% | 41.67% | 41.50% | 8.50 pp | -43 | 20 | -2.15 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| BTC Market Hours Daily | lstm | LSTM | 253 | 103 | 150 | 40.71% | 41.25% | 40.71% | 9.29 pp | -47 | 21 | -2.24 |
| BTC Hourly | transformer | Transformer | 229 | 101 | 128 | 44.10% | 44.10% | 44.10% | 5.90 pp | -27 | 10 | -2.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Hourly | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |
| BTC Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 10 | -3.90 |
| BTC Daily | transformer | Transformer | 255 | 103 | 152 | 40.39% | 39.58% | 40.39% | 9.61 pp | -49 | 11 | -4.45 |
| BTC Hourly | lstm | LSTM | 229 | 86 | 143 | 37.55% | 37.55% | 37.55% | 12.45 pp | -57 | 10 | -5.70 |
| BTC Daily | rf | RandomForest | 255 | 96 | 159 | 37.65% | 36.67% | 37.65% | 12.35 pp | -63 | 11 | -5.73 |
| BTC Daily | xgb | XGBoost | 265 | 94 | 171 | 35.47% | 35.00% | 35.47% | 14.53 pp | -77 | 12 | -6.42 |
| BTC Hourly | xgb | XGBoost | 229 | 80 | 149 | 34.93% | 34.93% | 34.93% | 15.07 pp | -69 | 10 | -6.90 |
| BTC Daily | lstm | LSTM | 255 | 88 | 167 | 34.51% | 35.00% | 34.51% | 15.49 pp | -79 | 11 | -7.18 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 229 | 115 | 114 | 50.22% | 50.22% | 50.22% | 0.22 pp | 1 | 10 | 0.10 |
| BTC Hourly | transformer | Transformer | 229 | 101 | 128 | 44.10% | 44.10% | 44.10% | 5.90 pp | -27 | 10 | -2.70 |
| BTC Hourly | nn | NN | 229 | 96 | 133 | 41.92% | 41.92% | 41.92% | 8.08 pp | -37 | 10 | -3.70 |
| BTC Hourly | rf | RandomForest | 229 | 95 | 134 | 41.48% | 41.48% | 41.48% | 8.52 pp | -39 | 10 | -3.90 |
| BTC Hourly | lstm | LSTM | 229 | 86 | 143 | 37.55% | 37.55% | 37.55% | 12.45 pp | -57 | 10 | -5.70 |
| BTC Hourly | xgb | XGBoost | 229 | 80 | 149 | 34.93% | 34.93% | 34.93% | 15.07 pp | -69 | 10 | -6.90 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 255 | 119 | 136 | 46.67% | 46.67% | 46.67% | 3.33 pp | -17 | 11 | -1.55 |
| BTC Daily | nn | NN | 255 | 116 | 139 | 45.49% | 44.17% | 45.49% | 4.51 pp | -23 | 11 | -2.09 |
| BTC Daily | transformer | Transformer | 255 | 103 | 152 | 40.39% | 39.58% | 40.39% | 9.61 pp | -49 | 11 | -4.45 |
| BTC Daily | rf | RandomForest | 255 | 96 | 159 | 37.65% | 36.67% | 37.65% | 12.35 pp | -63 | 11 | -5.73 |
| BTC Daily | xgb | XGBoost | 265 | 94 | 171 | 35.47% | 35.00% | 35.47% | 14.53 pp | -77 | 12 | -6.42 |
| BTC Daily | lstm | LSTM | 255 | 88 | 167 | 34.51% | 35.00% | 34.51% | 15.49 pp | -79 | 11 | -7.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 253 | 132 | 121 | 52.17% | 52.08% | 52.17% | 2.17 pp | 11 | 20 | 0.55 |
| BTC Market Hours | transformer | Transformer | 253 | 121 | 132 | 47.83% | 48.33% | 47.83% | 2.17 pp | -11 | 20 | -0.55 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 253 | 118 | 135 | 46.64% | 46.25% | 46.64% | 3.36 pp | -17 | 20 | -0.85 |
| BTC Market Hours | xgb | XGBoost | 253 | 118 | 135 | 46.64% | 45.83% | 46.64% | 3.36 pp | -17 | 20 | -0.85 |
| BTC Market Hours | rf | RandomForest | 253 | 113 | 140 | 44.66% | 44.17% | 44.66% | 5.34 pp | -27 | 20 | -1.35 |
| BTC Market Hours | lstm | LSTM | 253 | 105 | 148 | 41.50% | 41.67% | 41.50% | 8.50 pp | -43 | 20 | -2.15 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 253 | 126 | 127 | 49.80% | 50.00% | 49.80% | 0.20 pp | -1 | 21 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 253 | 125 | 128 | 49.41% | 48.75% | 49.41% | 0.59 pp | -3 | 21 | -0.14 |
| BTC Market Hours Daily | nn | NN | 253 | 122 | 131 | 48.22% | 47.92% | 48.22% | 1.78 pp | -9 | 21 | -0.43 |
| BTC Market Hours Daily | xgb | XGBoost | 253 | 112 | 141 | 44.27% | 44.58% | 44.27% | 5.73 pp | -29 | 21 | -1.38 |
| BTC Market Hours Daily | rf | RandomForest | 253 | 110 | 143 | 43.48% | 42.92% | 43.48% | 6.52 pp | -33 | 21 | -1.57 |
| BTC Market Hours Daily | lstm | LSTM | 253 | 103 | 150 | 40.71% | 41.25% | 40.71% | 9.29 pp | -47 | 21 | -2.24 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Hourly | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Hourly | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Hourly | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Hourly | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 14 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 221 | 103 | 118 | 46.61% | 46.61% | 46.61% | 3.39 pp | -15 | 14 | -1.07 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 14 | -1.21 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 221 | 97 | 124 | 43.89% | 43.89% | 43.89% | 6.11 pp | -27 | 14 | -1.93 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 221 | 96 | 125 | 43.44% | 43.44% | 43.44% | 6.56 pp | -29 | 14 | -2.07 |
| Consolidated Daily/Hourly Refresh | nn | NN | 221 | 90 | 131 | 40.72% | 40.72% | 40.72% | 9.28 pp | -41 | 14 | -2.93 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 75 | 33 | 42 | 44.00% | 44.00% | 44.00% | 6.00 pp | -9 | 6 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | transformer | Transformer | 75 | 32 | 43 | 42.67% | 42.67% | 42.67% | 7.33 pp | -11 | 6 | -1.83 |
| Consolidated Market Hours Daily | lstm | LSTM | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 6 | -2.17 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 75 | 29 | 46 | 38.67% | 38.67% | 38.67% | 11.33 pp | -17 | 6 | -2.83 |
| Consolidated Market Hours Daily | nn | NN | 75 | 28 | 47 | 37.33% | 37.33% | 37.33% | 12.67 pp | -19 | 6 | -3.17 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
